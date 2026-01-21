#include <torch/torch.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <iostream>


#include <thread>

#include "mcts_agent_parallel.h"
#include "logger.h"

Mcts_agent_parallel::Mcts_agent_parallel(double exploration_factor,
                         int number_iteration,
                         LogLevel log_level,
                         float temperature,
                         float dirichlet_alpha,
                         float dirichlet_epsilon,
                         std::shared_ptr<AlphaZModel> network,
                         int max_depth,
                         bool tree_reuse,
                         float virtual_loss,
                         int num_workers,
                         int nn_batch_size)
    : exploration_factor(exploration_factor),
      number_iteration(number_iteration),
      logger(Logger::instance(log_level)),
      temperature(temperature),
      dirichlet_alpha(dirichlet_alpha),
      dirichlet_epsilon(dirichlet_epsilon),
      network(network),
      max_depth(max_depth),
      random_generator(std::random_device{}()),
      tree_reuse(tree_reuse),
      virtual_loss_value(virtual_loss),
      num_workers(num_workers),
      nn_batch_size(nn_batch_size) {

        torch::Device device= torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        torch::Device device_cpu =  torch::kCPU;
        const int pool_size = 64;
        tensor_pool = std::make_unique<TensorPool>(device_cpu, pool_size);
        network->eval();
       }

Mcts_agent_parallel::Node::Node(Cell_state player, std::array<int, 4> move, float prior_proba, float value_from_nn,
                       std::shared_ptr<Node> parent_node)
    : value_from_nn(value_from_nn),
      value_from_mcts(0.0f),
      expanded(false),
      acc_value(0.0f),
      prior_proba(prior_proba),
      visit_count(0),
      move(move),
      player(player),
      child_nodes(),
      parent_node(parent_node) {}


std::pair<std::array<int, 4>, torch::Tensor> Mcts_agent_parallel::choose_move(const Board& board, Cell_state player) {

    logger->log_mcts_start(player);
    // // Create a new root node and expand it
    std::array<int, 4> arr = {-1, -1, -1, -1};
    root = std::make_shared<Node>(player, arr, 0.0, 0.0, nullptr);

    // Initialize root with Dirichlet noise for exploration
    initiate_and_run_nn(root, board, true, dirichlet_alpha, dirichlet_epsilon);

    int mcts_iteration_counter = 0;


    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    perform_mcts_iterations_parallel(number_iteration, mcts_iteration_counter, board);
    // perform_mcts_iterations(number_iteration, mcts_iteration_counter, board);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Output the time
    std::cout << "MCTS iterations took: " << duration.count() << " micro s" << std::endl;



    logger->log_timer_ran_out(mcts_iteration_counter);
    logger->log_root_stats(root->visit_count, root->child_nodes.size());

    auto [best_child, policy_from_mcts] = sample_child_and_get_policy(root);

    logger->log_best_child_chosen(mcts_iteration_counter, best_child->move,
                                    best_child->value_from_mcts, best_child->visit_count);
    logger->log_mcts_end();

    for (const auto& child : root->child_nodes) {
        float avg_value = (child->visit_count > 0) ? 
                            child->acc_value / child->visit_count : 0.0f;
        logger->log_child_node_stats(child->move, child->acc_value,
                                        child->visit_count, child->prior_proba);
    }

    return {best_child->move, policy_from_mcts};

}

std::vector<float> Mcts_agent_parallel::generate_dirichlet_noise(int num_moves, float alpha) {
    if (num_moves == 0) return {};

    std::gamma_distribution<float> gamma_dist(alpha, 1.0);
    std::vector<float> noise(num_moves);
    float noise_sum = 0.0;

    // Generate gamma samples
    for (int i = 0; i < num_moves; ++i) {
        noise[i] = gamma_dist(random_generator);
        noise_sum += noise[i];
    }

    // Normalize to sum to 1
    for (int i = 0; i < num_moves; ++i) {
        noise[i] /= noise_sum;
    }
    return noise;
}

float Mcts_agent_parallel::initiate_and_run_nn(const std::shared_ptr<Node>& node, const Board& board,
                                      bool add_dirichlet_noise = false, float dirichlet_alpha = 0.3,
                                      float exploration_fraction = 0.25) {
    Cell_state current_player = node->player;
    Cell_state actual_player = node->player;

    auto [input, legal_mask] = tensor_pool->acquire();
    
    board.fill_tensor(input, current_player);
    board.fill_mask(legal_mask, current_player);
    
    input = input.unsqueeze(0);
    legal_mask = legal_mask.unsqueeze(0);
    auto [policy, value] = network->predict(input, legal_mask);
    tensor_pool->release(input, legal_mask);

    std::vector<std::pair<std::array<int, 4>, float>> move_with_logit = get_moves_with_probs(policy);
    
    logger->log_nn_evaluation(node->move, value.item<float>(), move_with_logit.size());

    std::vector<float> noise;
    if (add_dirichlet_noise && !move_with_logit.empty()) {
        noise = generate_dirichlet_noise(move_with_logit.size(), dirichlet_alpha);

        for (size_t i = 0; i < move_with_logit.size(); ++i) {
            move_with_logit[i].second =
                (1.0 - exploration_fraction) * move_with_logit[i].second + exploration_fraction * noise[i];
        }

        logger->log_dirichlet_noise_applied(dirichlet_alpha, exploration_fraction);
    }

    // For each valid move, initialize a new child node
    int idx = 0;
    for (const auto& [move, logit] : move_with_logit) {
        if (move[3] < 1) {
            actual_player = (current_player == Cell_state::X ? Cell_state::O : Cell_state::X);
        } else {
            actual_player = current_player;
        }

        std::shared_ptr<Node> new_child =
            std::make_shared<Node>(actual_player, std::array<int, 4>(move), logit, 0.0, node);
        node->child_nodes.push_back(new_child);
        idx++;
    }

    logger->log_expansion(node->move, node->child_nodes.size());
    node->value_from_nn = value.item<float>();
    node->expanded = true;

    return value.item<float>();
}

std::pair<std::shared_ptr<Mcts_agent_parallel::Node>, torch::Tensor> Mcts_agent_parallel::sample_child_and_get_policy(
    const std::shared_ptr<Node>& parent_node) const {
    
    const int X = 4;
    const int Y = 4;
    const int DIR = 1;
    const int TAR = 1; 
    const int total_size = X * Y * DIR * TAR;
    
    auto index = [&](int x, int y, int dir, int tar) -> int {
        int dir_idx = dir - 1; 
        int tar_idx = tar + 1; 
        return x * (Y * DIR * TAR) + y * (DIR * TAR) + dir_idx * TAR + tar_idx;
    };
    
    const auto& children = parent_node->child_nodes;
    const int num_legal_moves = children.size();
    
    // Only allocate for legal moves
    std::vector<int> legal_indices;
    std::vector<float> adjusted_visits;
    legal_indices.reserve(num_legal_moves);
    adjusted_visits.reserve(num_legal_moves);
    
    // Temperature = 0: Select best child deterministically
    if (temperature == 0.0f) {
        std::shared_ptr<Node> best_child = nullptr;
        int best_idx = -1;
        int max_visits = -1;
        
        for (const auto& child : children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
                best_child = child;
                best_idx = index(child->move[0], child->move[1], child->move[2], child->move[3]);
            }
        }
        
        torch::Tensor policy = torch::zeros({total_size}, torch::kFloat32);
        policy[best_idx] = 1.0f;
        
        return {best_child, policy};
    }
    
    // Temperature > 0
    float sum = 0.0f;
    for (const auto& child : children) {
        int idx = index(child->move[0], child->move[1], child->move[2], child->move[3]);
        float adjusted = std::pow(static_cast<float>(child->visit_count), 1.0f / temperature);
        
        legal_indices.push_back(idx);
        adjusted_visits.push_back(adjusted);
        sum += adjusted;
    }
    
    // Normalize
    torch::Tensor policy = torch::zeros({total_size}, torch::kFloat32);
    auto policy_acc = policy.accessor<float, 1>();
    
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < num_legal_moves; ++i) {
        policy_acc[legal_indices[i]] = adjusted_visits[i] * inv_sum;
    }
    
    // Sample from legal moves only
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(adjusted_visits.begin(), adjusted_visits.end());
    int selected_child_idx = dist(gen);
    
    return {children[selected_child_idx], policy};
}

std::vector<std::pair<std::array<int, 4>, float>> Mcts_agent_parallel::get_moves_with_probs(
    const torch::Tensor& log_probs_tensor) const {  
    
    const int X = 4;
    const int Y = 4;
    const int DIR = 1;
    const int TAR = 1;
    const int total = X * Y * DIR * TAR;
    
    torch::Tensor probs = torch::exp(log_probs_tensor).to(torch::kCPU).contiguous().view({total});
    const float* data = probs.data_ptr<float>();
    
    std::vector<std::pair<std::array<int, 4>, float>> moves;
    moves.reserve(total);
    
    float sum = 0.0f;
    
    for (int idx = 0; idx < total; idx++) {
        float p = data[idx];
        
        if (p <= 0.0f) continue;
         
        int t = idx;
        int x = t / (Y * DIR * TAR);
        t %= (Y * DIR * TAR);
        int y = t / (DIR * TAR);
        t %= (DIR * TAR);
        int dir = (t / TAR) + 1;
        int tar = (t % TAR) - 1;
        
        std::array<int, 4> move = {x, y, dir, tar};
        moves.emplace_back(move, p);
        sum += p;
    }
    
    
    if (moves.empty()) {
        return moves;
    }
    
    //Normalize
    for (size_t i = 0; i < moves.size(); i++) {
        moves[i].second /= sum;
    }

    
    return moves;
}


void Mcts_agent_parallel::backpropagate_remove_virtual_loss(
    std::shared_ptr<Node>& node, float value) {
    
    std::shared_ptr<Node> current_node = node;
    
    while (current_node != nullptr) {
        std::lock_guard<std::mutex> lock(current_node->node_mutex);
        
        // Remove virtual loss
        current_node->virtual_loss -= virtual_loss_value;
        current_node->virtual_visit_count.fetch_sub(1);
        
        // Negate because node stores parent's perspective but value is from node's perspective
        value = -value;
        
        // Update real statistics
        current_node->acc_value += value;
        current_node->visit_count += 1;
        current_node->value_from_mcts = current_node->acc_value / current_node->visit_count;
        
        current_node = current_node->parent_node;
    }
}

std::pair<std::shared_ptr<Mcts_agent_parallel::Node>, Board>
Mcts_agent_parallel::select_with_virtual_loss(
    const std::shared_ptr<Node>& parent_node,
    Board board)

{
    std::shared_ptr<Node> current = parent_node;
    Cell_state current_player = current->player;
    std::atomic<int> lock_wait_count{0};
    std::atomic<int> lock_acquire_count{0};

    while (current->expanded && !current->child_nodes.empty()) 
    {
        std::shared_ptr<Node> best_child = nullptr;
        double max_score = -std::numeric_limits<double>::infinity();
        
        {


            // In select_with_virtual_loss:
            auto lock_start = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(current->node_mutex);
            auto lock_end = std::chrono::high_resolution_clock::now();

            if (std::chrono::duration_cast<std::chrono::microseconds>(lock_end - lock_start).count() > 100) {
                lock_wait_count++;
            }
            lock_acquire_count++;
            
            for (auto& child : current->child_nodes) {
                // Calculate Q-value with virtual loss
                double q_value = 0.0;
                int total_visits = child->visit_count + child->virtual_visit_count.load();
                
                if (total_visits > 0) {
                    float total_value = child->acc_value - child->virtual_loss;
                    q_value = total_value / total_visits;
                }
                
                // Calculate exploration term
                int parent_total_visits = current->visit_count + current->virtual_visit_count.load();
                double u_value = exploration_factor * child->prior_proba * 
                    std::sqrt(parent_total_visits) / (1 + total_visits);
                
                double score = q_value + u_value;
                
                if (score > max_score) {
                    max_score = score;
                    best_child = child;
                }
            }
            
            // Add virtual loss IMMEDIATELY to discourage other workers
            if (best_child) {
                best_child->virtual_loss += virtual_loss_value;
                best_child->virtual_visit_count.fetch_add(1);
            }
        }

        if (!best_child) break;

        // Apply move
        board.make_move(best_child->move[0], best_child->move[1], 
                       best_child->move[2], best_child->move[3], current_player);

        if (best_child->move[3] < 1) {
            current_player = (current_player == Cell_state::X ? Cell_state::O : Cell_state::X);
            board.clear_state();
        }

        current = best_child;
    }


    return { current, board };
}


void Mcts_agent_parallel::expand_node_from_policy(
    const std::shared_ptr<Node>& node,
    const Board& board,
    const torch::Tensor& policy,
    float value)
{
    
    Cell_state current_player = node->player;
    Cell_state actual_player = node->player;

    
    std::vector<std::pair<std::array<int,4>, float>> move_with_logit = 
        get_moves_with_probs(policy);
    
    std::lock_guard<std::mutex> lock(node->node_mutex);
    

    // For each valid move, create a new child node
    int idx = 0;
    for (const auto& [move, logit] : move_with_logit) {
        
        if (move[3] < 1) {
            actual_player = (current_player == Cell_state::X ? Cell_state::O : Cell_state::X);
        } else {
            actual_player = current_player;
        }

        std::shared_ptr<Node> new_child =
            std::make_shared<Node>(actual_player, std::array<int, 4>(move), 
                                    logit, 0.0, node);
        
        node->child_nodes.push_back(new_child);
        idx++;
    }
    node->value_from_nn = value;
    node->expanded = true;
    
}



void Mcts_agent_parallel::perform_mcts_iterations_parallel(
    int number_iteration,
    int& mcts_iteration_counter, 
    const Board& board) 
{
    std::atomic<int> global_counter(0);
    std::atomic<bool> should_stop(false);
    std::atomic<long long> time_in_selection{0};
    std::atomic<long long> time_in_nn{0};
    std::atomic<long long> time_in_backprop{0};
    
    // Shared queue for batching NN inference
    struct BatchItem {
        std::shared_ptr<Node> node;
        Board board_state;
    };
    std::vector<BatchItem> batch_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    torch::Tensor batched_input = torch::empty(
        {nn_batch_size, 3, 4, 4}, 
        torch::TensorOptions().device(current_device).dtype(torch::kFloat32));
    torch::Tensor batched_masks = torch::empty(
        {nn_batch_size, 16}, 
        torch::TensorOptions().device(current_device).dtype(torch::kFloat32));

    // Inference thread (dedicated to batch NN calls)
    std::thread inference_thread([&]() {
        while (!should_stop.load()) {
            std::vector<BatchItem> batch;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                // Wait until we have a batch or timeout
                queue_cv.wait_for(lock, std::chrono::microseconds(200), [&] {
                    return batch_queue.size() >= nn_batch_size || should_stop.load();
                });
                
                if (batch_queue.empty()) continue;
                
                // Take up to nn_batch_size items
                size_t take_count = std::min(batch_queue.size(), static_cast<size_t>(nn_batch_size));
                batch.reserve(take_count);
                for (size_t i = 0; i < take_count; i++) {
                    batch.push_back(std::move(batch_queue[i]));
                }
                batch_queue.erase(batch_queue.begin(), batch_queue.begin() + take_count);
            }
            
            if (batch.empty()) continue;
            
            torch::NoGradGuard no_grad;

            // BATCH NN INFERENCE
            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> masks;
            inputs.reserve(batch.size());
            masks.reserve(batch.size());
            
            for (auto& item : batch) {
                    auto [board_tensor, mask_tensor] = tensor_pool->acquire();
                    
                    // Fill directly into pre-allocated GPU tensors
                    item.board_state.fill_tensor(board_tensor, item.node->player);
                    item.board_state.fill_mask(mask_tensor, item.node->player);
                    
                    inputs.push_back(board_tensor);
                    masks.push_back(mask_tensor);
                }
            
            
            torch::Tensor batched_input = torch::stack(inputs).to(current_device);
            torch::Tensor batched_masks = torch::stack(masks).to(current_device);

            // Predict
            auto [policies, values] = network->predict(batched_input, batched_masks);
                
            // Expand nodes and signal completion
            for (size_t i = 0; i < batch.size(); i++) {
                auto& item = batch[i];
                float value = values[i].item<float>();
                
                // Expand the node with policy
                expand_node_from_policy(item.node, item.board_state, policies[i], value);
                
                // Signal this node is ready for backprop
                {
                    std::lock_guard<std::mutex> lock(item.node->node_mutex);
                    item.node->value_from_nn = value;
                    item.node->nn_evaluated = true;
                }
                item.node->cv.notify_all();
            }

            for (size_t i = 0; i < batch.size(); i++) {
                tensor_pool->release(inputs[i], masks[i]);
            }
        }
    });

    
    // Worker threads (selection + backprop)
    std::vector<std::thread> workers;
    for (int t = 0; t < num_workers; t++) {
        workers.emplace_back([&]() {
            while (true) {
                int current_iteration = global_counter.fetch_add(1);
                if (current_iteration >= number_iteration) break;
                
                // STEP 1: Selection with VIRTUAL LOSS
                auto t1 = std::chrono::high_resolution_clock::now();
                auto [leaf_node, board_state] = select_with_virtual_loss(root, board);
                auto t2 = std::chrono::high_resolution_clock::now();
                
                // STEP 2: Check if we need NN evaluation
                float value_to_backprop;
                
                if (!leaf_node->expanded) {
                    // Check for immediate win/loss
                    Cell_state winner = board_state.check_winner();
                    
                    if (winner == leaf_node->player) {
                        value_to_backprop = 1.0f;
                        {
                            std::lock_guard<std::mutex> lock(leaf_node->node_mutex);
                            leaf_node->value_from_nn = value_to_backprop;
                            leaf_node->nn_evaluated = true;
                            leaf_node->expanded = true;
                        }
                    }
                    else if (winner != Cell_state::Empty) {
                        value_to_backprop = -1.0f;
                        {
                            std::lock_guard<std::mutex> lock(leaf_node->node_mutex);
                            leaf_node->value_from_nn = value_to_backprop;
                            leaf_node->nn_evaluated = true;
                            leaf_node->expanded = true;
                        }
                    }
                    else {
                        // Add to batch queue for NN evaluation
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            batch_queue.push_back({leaf_node, board_state});
                        }
                        queue_cv.notify_one();
                        
                        // STEP 3: WAIT for NN evaluation
                        {
                            std::unique_lock<std::mutex> lock(leaf_node->node_mutex);
                            leaf_node->cv.wait(lock, [&] { 
                                return leaf_node->nn_evaluated.load(); 
                            });
                        }
                        
                        value_to_backprop = leaf_node->value_from_nn;
                    }
                }
                else {
                    // Node already expanded, use stored value
                    value_to_backprop = leaf_node->value_from_nn;
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                // STEP 4: Backpropagate (remove virtual loss)
                backpropagate_remove_virtual_loss(leaf_node, value_to_backprop);
                auto t4 = std::chrono::high_resolution_clock::now();
                // logger->log_simulation_end(value_to_backprop);
                
                // Worker immediately starts next iteration

                time_in_selection += std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
                time_in_nn += std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
                time_in_backprop += std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
            }
        });
    }
    
    // Wait for all workers to complete
    for (auto& w : workers) {
        w.join();
    }
    
    // Stop inference thread
    should_stop.store(true);
    queue_cv.notify_all();
    inference_thread.join();
    
    mcts_iteration_counter = number_iteration;

}
