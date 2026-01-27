#include "game.h"
#include "logger.h"
#include <torch/torch.h>
#include <chrono>
#include "alphaz_model.h"
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <iostream>
#include <random>


static std::mt19937 random_generator(std::random_device{}());

Game::Game(int board_size, std::unique_ptr<Player> player1,
           std::unique_ptr<Player> player_2, GameDataset& dataset, bool evaluation, Cell_state player_to_evaluate)
    : board(board_size), current_player_index(0), dataset_(dataset), evaluation(evaluation), player_to_evaluate(player_to_evaluate) {
  players[0] = std::move(player1);
  players[1] = std::move(player_2);

  result_z.clear();
  optimal_moves = 0;
  eval_moves = 0;
}

Cell_state Game::play() {

    int move_counter = 0;
    int max_move = 70;

    result_z.clear();

    Cell_state current_player =
            current_player_index == 0 ? Cell_state::X : Cell_state::O;

    while (board.check_winner() == Cell_state::Empty) {

        if (move_counter>max_move){
            break;
        }
        auto valid_moves = board.get_valid_moves(current_player);
        if (valid_moves.empty()) {
            break;
        }

        Cell_state current_player =
            current_player_index == 0 ? Cell_state::X : Cell_state::O;
        
        if (!evaluation){
            auto* mcts = dynamic_cast<Mcts_player_selfplay*>(players[current_player_index].get());
            if (move_counter < 6) {
                mcts->set_temperature(1.0);
            } else {
                mcts->set_temperature(0.1); // Rest: almost deterministic
            }
        }

        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);
        
        
        if (!evaluation){
            //Collect data
            auto board_tensor = board.to_tensor(current_player);
            auto pi_tensor = logits;
            auto mask_tensor = board.get_legal_mask(current_player);
            
            float z_value;
            if (current_player == Cell_state::X) {z_value = 0.0;} 
            else if (current_player == Cell_state::O) {z_value = 1.0;}

            auto z_tensor = torch::tensor(z_value, torch::dtype(torch::kFloat32));
            result_z.push_back(z_tensor);

            dataset_.add_position(board_tensor, pi_tensor, z_tensor, mask_tensor);
        }

        //TOP 1 ACCURACY
        else if (evaluation && move_counter >= 4 && current_player==player_to_evaluate) {
            auto [chosen_move, minimax_logits] = minimax_agent->choose_move(board, current_player);
            auto mask = board.get_legal_mask(current_player);
            bool is_optimal = is_policy_optimal(logits, minimax_logits,mask);
            // std::cout << "Policy chose optimal move: " << (is_optimal ? "YES" : "NO") << "\n";
            eval_moves++;
            if (is_optimal) {
                optimal_moves++;
            }
        }

        int chosen_x = chosen_move[0];
        int chosen_y = chosen_move[1];
        int chosen_dir = chosen_move[2];
        int chosen_tar = chosen_move[3];

        board.make_move(chosen_x, chosen_y, chosen_dir, chosen_tar, current_player);
        switch_player();

        move_counter ++;
    }

    Cell_state winner = board.check_winner();
    //update the z targets on the data based on the winner
    if (!evaluation){
        dataset_.update_z(result_z, winner);
    }
    // board.display_board(std::cout);
    return winner;
}

Cell_state Game::simple_play() {

    int move_counter = 0;;

    Cell_state current_player =
            current_player_index == 0 ? Cell_state::X : Cell_state::O;

    while (board.check_winner() == Cell_state::Empty) {

        auto valid_moves = board.get_valid_moves(current_player);
        if (valid_moves.empty()) {
            break;
        }

        Cell_state current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;

        std::cout << "\nPlayer " << current_player_index + 1 << "'s turn:" << std::endl << std::endl;
        board.display_board(std::cout);
        
        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);

        int chosen_x = chosen_move[0];
        int chosen_y = chosen_move[1];
        int chosen_dir = chosen_move[2];
        int chosen_tar = chosen_move[3];
        
        std::cout << "\nPlayer " << current_player_index + 1 << " chose move: " << print_move(chosen_move) << std::endl;
        
        board.make_move(chosen_x, chosen_y, chosen_dir, chosen_tar, current_player);

        switch_player();

        move_counter ++;
    }
    Cell_state winner = board.check_winner();

    board.display_board(std::cout);
    std::cout << "Player " << winner << " wins!" << std::endl;
    
    return winner;
}


std::string Game::print_move(std::array<int, 4> move) {

    char column = 'a' + move[1];
    int row = move[0] + 1;

    std::string message = "(" + std::to_string(row) + ", " + column + ") ";
    return message;
}

void Game::switch_player() {
    current_player_index = 1 - current_player_index;
}


void Game::random_move(int random_move_number) {

    int move_counter = 0;
    Cell_state player = (current_player_index == 0 
                         ? Cell_state::X 
                         : Cell_state::O);

    while (move_counter < random_move_number) {


        std::vector<std::array<int, 4>> valid_moves = board.get_valid_moves(player);
        if (valid_moves.empty()) {
            break;
        }

        std::uniform_int_distribution<> dist(0, static_cast<int>(valid_moves.size() - 1));
        const std::array<int, 4>& random_move = valid_moves[dist(random_generator)];

        board.make_move(random_move[0], random_move[1],
                        random_move[2], random_move[3], player);

        move_counter++;

        if (board.check_winner() != Cell_state::Empty) {
            break;
        }

        if (random_move[3] < 1) {
            switch_player();
            player = (current_player_index == 0 
                       ? Cell_state::X 
                       : Cell_state::O);

            board.clear_state();
        }
    }
}

bool Game::is_policy_optimal(const torch::Tensor& policy_logits, 
                       const torch::Tensor& minimax_logits,
                       const torch::Tensor& legal_mask) {
    auto policy_flat = policy_logits.flatten();
    auto minimax_flat = minimax_logits.flatten();
    auto mask_flat = legal_mask.flatten();
    
    auto policy_acc = policy_flat.accessor<float, 1>();
    auto minimax_acc = minimax_flat.accessor<float, 1>();
    auto mask_acc = mask_flat.accessor<float, 1>();
    
    const float EPSILON = 1e-6;
    
    // Find policy's top-1 move (x, y) - only among legal moves
    int policy_best_idx = -1;
    float policy_max_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < policy_flat.size(0); ++i) {
        if (mask_acc[i] > 0.5f && policy_acc[i] > policy_max_score) {
            policy_max_score = policy_acc[i];
            policy_best_idx = i;
        }
    }
    
    if (policy_best_idx == -1) {
        std::cout << "No legal moves found!\n";
        return false;
    }
    
    int policy_x = policy_best_idx / 4;
    int policy_y = policy_best_idx % 4;
    
    // Find all policy moves with top score (ties) - only legal
    std::set<std::pair<int, int>> policy_optimal_moves;
    for (int i = 0; i < policy_flat.size(0); ++i) {
        if (mask_acc[i] > 0.5f && std::abs(policy_acc[i] - policy_max_score) < EPSILON) {
            int x = i / 4;
            int y = i % 4;
            policy_optimal_moves.insert({x, y});
        }
    }
    
    // Find all minimax optimal moves (all with max score) - only legal
    float minimax_max_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < minimax_flat.size(0); ++i) {
        if (mask_acc[i] > 0.5f && minimax_acc[i] > minimax_max_score) {
            minimax_max_score = minimax_acc[i];
        }
    }
    
    std::set<std::pair<int, int>> minimax_optimal_moves;
    for (int i = 0; i < minimax_flat.size(0); ++i) {
        if (mask_acc[i] > 0.5f && std::abs(minimax_acc[i] - minimax_max_score) < EPSILON) {
            int x = i / 4;
            int y = i % 4;
            minimax_optimal_moves.insert({x, y});
        }
    }
    
    // // Print policy's top-1 move(s)
    // std::cout << "Policy Top-1 Move(s) (score: " << policy_max_score << "): ";
    // for (const auto& [x, y] : policy_optimal_moves) {
    //     std::cout << "(" << x << "," << y << ") ";
    // }
    // std::cout << "\n";
    
    // // Print minimax's optimal move(s)
    // std::cout << "Minimax Optimal Move(s) (score: " << minimax_max_score << "): ";
    // for (const auto& [x, y] : minimax_optimal_moves) {
    //     std::cout << "(" << x << "," << y << ") ";
    // }
    // std::cout << "\n";
    
    // // Check if policy's top-1 is in minimax's optimal set
    bool is_optimal = minimax_optimal_moves.count({policy_x, policy_y}) > 0;
    // std::cout << "Policy chose optimal: " << (is_optimal ? "YES" : "NO") << "\n\n";
    
    return is_optimal;
}