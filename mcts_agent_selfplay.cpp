#include <torch/torch.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <random>


#include "mcts_agent_selfplay.h"
#include "inference_queue.h"
#include "logger.h"



Mcts_agent_selfplay::Mcts_agent_selfplay(double exploration_factor,
                         int number_iteration,
                         LogLevel log_level,
                         float temperature,
                         float dirichlet_alpha,
                         float dirichlet_epsilon,
                         InferenceQueue* inference_queue,
                         int max_depth,
                         bool tree_reuse)
    : exploration_factor(exploration_factor),
      number_iteration(number_iteration),
      logger(Logger::instance(log_level)),
      temperature(temperature),
      dirichlet_alpha(dirichlet_alpha),
      dirichlet_epsilon(dirichlet_epsilon),
      inference_queue(inference_queue),
      max_depth(max_depth),
      random_generator(std::random_device{}()),
      tree_reuse(tree_reuse) {
      }

Mcts_agent_selfplay::Node::Node(Cell_state player, std::array<int, 4> move, float prior_proba, float value_from_nn,
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

std::vector<float> Mcts_agent_selfplay::generate_dirichlet_noise(int num_moves, float alpha) {
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


std::pair<std::array<int, 4>, torch::Tensor> Mcts_agent_selfplay::choose_move(const Board& board, Cell_state player) {

    logger->log_mcts_start(player);
    // // Create a new root node and expand it
    std::array<int, 4> arr = {-1, -1, -1, -1};
    root = std::make_shared<Node>(player, arr, 0.0, 0.0, nullptr);

    // Initialize root with Dirichlet noise for exploration
    initiate_and_run_nn(root, board, true, dirichlet_alpha, dirichlet_epsilon);

    int mcts_iteration_counter = 0;

    // Run MCTS until the timer runs out
    perform_mcts_iterations(number_iteration, mcts_iteration_counter, board);

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

float Mcts_agent_selfplay::initiate_and_run_nn(const std::shared_ptr<Node>& node, const Board& board,
                                      bool add_dirichlet_noise = false, float dirichlet_alpha = 0.4,
                                      float exploration_fraction = 0.25) {
    Cell_state current_player = node->player;
    Cell_state actual_player = node->player;

    torch::Tensor input = board.to_tensor(current_player);
    torch::Tensor legal_mask = board.get_legal_mask(current_player);
    
    auto [policy, value] = inference_queue->evaluate_and_wait(input, legal_mask);

    std::vector<std::pair<std::array<int, 4>, float>> move_with_logit = get_moves_with_probs(policy);
    
    //Dive deeper
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


void Mcts_agent_selfplay::perform_mcts_iterations(int number_iteration, int& mcts_iteration_counter, const Board& board) {
    int count = 0;
    while (mcts_iteration_counter < number_iteration) {
        logger->log_iteration_number(mcts_iteration_counter + 1);

        logger->log_step("START SELECTION FROM", root->move);
        auto [chosen_child, new_board] = select_child_for_playout(root, board);
        logger->log_step("SELECTED", chosen_child->move);

        float value_from_nn = simulate_random_playout(chosen_child, new_board);

        logger->log_step("BACKPROPAGATION", chosen_child->move);
        backpropagate(chosen_child, value_from_nn);
        
        logger->log_step("FINAL STATS", chosen_child->move);
        for (const auto& child : root->child_nodes) {
            logger->log_child_node_stats(child->move, child->acc_value, child->visit_count, child->prior_proba);
        }
        mcts_iteration_counter++;
    }
}

std::pair<std::shared_ptr<Mcts_agent_selfplay::Node>, torch::Tensor> Mcts_agent_selfplay::sample_child_and_get_policy(
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

std::vector<std::pair<std::array<int, 4>, float>> Mcts_agent_selfplay::get_moves_with_probs(
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

    for (auto& m : moves) {
        m.second /= sum;
    }

    return moves;
}

std::pair<std::shared_ptr<Mcts_agent_selfplay::Node>, Board> Mcts_agent_selfplay::select_child_for_playout(
    const std::shared_ptr<Node>& parent_node, Board board) {
    std::shared_ptr<Node> current = parent_node;
    Cell_state current_player = current->player;

    while (current->expanded && !current->child_nodes.empty()) {
        // Pick best child
        std::shared_ptr<Node> best_child = current->child_nodes[0];
        double max_score = calculate_puct_score(best_child, current);

        for (size_t i = 0; i < current->child_nodes.size(); i++) {
            auto& child = current->child_nodes[i];
            double score = calculate_puct_score(child, current);

            float q_value = child->acc_value / child->visit_count;
            float u_value = score - q_value;
            logger->log_puct_details(child->move, q_value, u_value,
                                        child->prior_proba, child->visit_count,
                                        current->visit_count);
            if (score > max_score) {
                max_score = score;
                best_child = child;
            }
        }

        logger->log_selected_child(best_child->move, max_score);

        // Apply move
        board.make_move(best_child->move[0], best_child->move[1], best_child->move[2], best_child->move[3],
                        current_player);

        if (best_child->move[3] < 1) {
            // Switch player
            current_player = (current_player == Cell_state::X ? Cell_state::O : Cell_state::X);
            board.clear_state();
        }

        current = best_child;
    }

    return {current, board};
}

double Mcts_agent_selfplay::calculate_puct_score(const std::shared_ptr<Node>& child_node,
                                        const std::shared_ptr<Node>& parent_node) {
    return static_cast<double>(child_node->value_from_mcts +
                               exploration_factor * child_node->prior_proba *
                                   (std::sqrt(parent_node->visit_count) / (child_node->visit_count + 1)));
}

float Mcts_agent_selfplay::simulate_random_playout(const std::shared_ptr<Node>& node, Board board) {

    //The check here are from the actual node point of view.
    Cell_state winner = board.check_winner();
    if (winner == node->player) {
        logger->log_simulation_end(1.0);
        return 1.0;  // current player won

    } else if (winner == Cell_state::Empty) {
        float value = initiate_and_run_nn(node, board);
        logger->log_simulation_end(value);
        return value;
    } else {
        logger->log_simulation_end(-1.0);
        return -1.0;  // opponent won
    }
}

void Mcts_agent_selfplay::backpropagate(std::shared_ptr<Node>& node, float value) {

    std::shared_ptr<Node> current_node = node;
    while (current_node != nullptr) {
        
        // Lock the node's mutex before updating its data
        std::lock_guard<std::mutex> lock(current_node->node_mutex);

        // Negate because node stores parent's perspective but value is from node's perspective
        value = -value;

        current_node->acc_value += value;

        // Increment the node's visit count
        current_node->visit_count += 1;
        // Update accumulated value of the node
        
        current_node->value_from_mcts = current_node->acc_value / current_node->visit_count;

        logger->log_backpropagation_result(current_node->move, 
                                    current_node->acc_value,
                                    current_node->visit_count);

        // Move to the parent node for the next loop
        
        current_node = current_node->parent_node;
    }
}