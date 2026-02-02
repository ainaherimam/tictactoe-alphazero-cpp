#include "game.h"
#include <fstream>
#include <string>
#include <string>
#include <torch/torch.h>
#include <vector>

#include <random>
#include <chrono>

static std::mt19937 random_generator(std::random_device{}());

Game::Game(std::unique_ptr<Player> player_1,
           std::unique_ptr<Player> player_2, 
           PositionPool& pool,
           bool evaluation)
    : current_player_index(0), 
      pool_(pool),
      evaluation(evaluation), 
      current_game_id_(0),
      current_game_moves_(0)
{
  players[0] = std::move(player_1);
  players[1] = std::move(player_2);
  std::srand(static_cast<unsigned>(std::time(nullptr)));
}


void Game::collect_position(const float* policy,
                           uint8_t player_index,
                           uint16_t move_number) {
    Position& pos = pool_.acquire_position();
    
    Cell_state current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;

    board.to_float_array(current_player, pos.board.data());
    board.get_legal_mask(current_player, pos.mask.data());

    std::memcpy(pos.policy.data(), policy, POLICY_SIZE * sizeof(float));

    pos.z = 0.0f;
    pos.player_index = player_index;
}

Cell_state Game::play() {

    int max_move = 70;
    current_game_moves_ = 0;


    Cell_state current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;

    while (board.check_winner() == Cell_state::Empty) {
        if (current_game_moves_ > max_move) {
            break;
        }
        
        auto valid_moves = board.get_valid_moves(current_player);
        if (valid_moves.empty()) {
            break;
        }

        current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;
        
        if (!evaluation) {
            auto* mcts = dynamic_cast<Mcts_player_selfplay*>(players[current_player_index].get());
            if (current_game_moves_ < 6) {
                mcts->set_temperature(1.0);
            } else {
                mcts->set_temperature(0.1);
            }
        }
        
        // board.display_board(std::cout);

        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);
      
        if (!evaluation) {
            // Get raw float arrays instead of tensors
            // auto policy_data = logits.contiguous().data_ptr<float>();
            
            // Collect into pool instead of dataset
            collect_position(logits.data(),
                            static_cast<uint8_t>(current_player_index),
                            static_cast<uint16_t>(current_game_moves_));
        }

        // Update move history
        move_history += std::to_string(current_game_moves_ + 1) + "." 
           + char('0' + (4 - chosen_move.x)) 
           + char('a' + chosen_move.y) 
           + " ";


        board.make_move(chosen_move, current_player);
        switch_player();
        current_game_moves_++;
    }

    Cell_state winner = board.check_winner();
    
    if (!evaluation) {
        pool_.finalize_game(winner);
    }

    std::string p1_name = "Player 1";
    std::string p2_name = "Player 2";

    log_game("AZ Selfplay", p1_name, p2_name, winner, move_history);

    return winner;
}

void Game::switch_player() {
    current_player_index = 1 - current_player_index;
}

void Game::log_game(const std::string& event_name, 
                    const std::string& p1_name, 
                    const std::string& p2_name,
                    Cell_state winner,
                    std::string move_history) {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::tm* local_time = std::localtime(&time_t_now);
    std::string result =
    (winner == Cell_state::X) ? "1-0" :
    (winner == Cell_state::O) ? "0-1" :
                               "1/2-1/2";

    
    // Format: YYYY.MM.DD-HH.MM.SS.MM
    std::ostringstream timestamp;
    timestamp << std::setfill('0')
              << std::setw(4) << (local_time->tm_year + 1900) << "."
              << std::setw(2) << (local_time->tm_mon + 1) << "."
              << std::setw(2) << local_time->tm_mday << "-"
              << std::setw(2) << local_time->tm_hour << "."
              << std::setw(2) << local_time->tm_min << "."
              << std::setw(2) << local_time->tm_sec << "."
              << (std::rand() % 10000);
    
    // Build the log content
    std::string game_log;
    game_log = "[Event \"" + event_name + "\"]\n";
    game_log += "[Date \"" + timestamp.str() + "\"]\n";
    game_log += "[X \"" + p1_name + "\"]\n";
    game_log += "[O \"" + p2_name + "\"]\n";
    game_log += result + "\n";
    game_log += "\n" + move_history + "";
    
    
    // Create filename: event_name_YYYY.MM.DD-HH.MM.SS.pgn
    std::string filename = "games/" +event_name + "_" + timestamp.str() + ".pgn";
    
    // Save to file
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << game_log;
        outfile.close();
    }
}