#include "core/game/game.h"
#include "core/game/constants.h"
#include <string>
#include <string>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

static std::mt19937 random_generator(std::random_device{}());

Game::Game(std::unique_ptr<Player> player_1,
           std::unique_ptr<Player> player_2, 
           PositionPool& pool,
           bool is_evaluation)
    : current_player_index(0), 
      pool_(pool),
      is_evaluation(is_evaluation), 
      current_game_moves_(0)
{
  players[0] = std::move(player_1);
  players[1] = std::move(player_2);
}


void Game::collect_position(const float* policy,
                           uint8_t player_index,
                           uint16_t move_number) {
    Position& pos = pool_.acquire_position();

    Cell_state current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;
    
    static_assert(sizeof(pos.board) == INPUT_SIZE * sizeof(float),
                  "Position::board size mismatch — expected 3*16 floats");
    static_assert(sizeof(pos.policy) == BOARD_CELLS * sizeof(float),
                  "Position::policy size mismatch — expected 16 floats");
    static_assert(sizeof(pos.mask) == BOARD_CELLS * sizeof(float),
                  "Position::mask size mismatch — expected 16 floats");

    board.to_float_array(current_player, pos.board.data());
    board.get_legal_mask(current_player, pos.mask.data());
    std::memcpy(pos.policy.data(), policy, BOARD_CELLS * sizeof(float));

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
        
        if (!is_evaluation) {
            auto* mcts = dynamic_cast<Mcts_player_selfplay*>(players[current_player_index].get());
            if (current_game_moves_ < 6) {
                mcts->set_temperature(1.0);
            } else {
                mcts->set_temperature(0.1);
            }
        }
        
        // board.display_board(std::cout);

        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);

        if (!is_evaluation) {
            collect_position(logits.data(),
                            static_cast<uint8_t>(current_player_index),
                            static_cast<uint16_t>(current_game_moves_));
        }

        // Update string move history
        move_history += std::to_string(current_game_moves_ + 1) + "." 
           + char('0' + (4 - chosen_move.x)) 
           + char('a' + chosen_move.y) 
           + " ";


        board.make_move(chosen_move, current_player);
        switch_player();
        current_game_moves_++;
    }

    Cell_state winner = board.check_winner();
    
    if (!is_evaluation) {
        pool_.finalize_game(winner);
    }

    // board.display_board(std::cout);
    return winner;
}


Cell_state Game::play_via_api() {

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
        
        if (!is_evaluation) {
            auto* mcts = dynamic_cast<Mcts_player_triton*>(players[current_player_index].get());
            if (current_game_moves_ < 6) {
                mcts->set_temperature(1.0);
            } else {
                mcts->set_temperature(0.1);
            }
        }
        
        // board.display_board(std::cout);

        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);

        if (!is_evaluation) {
            collect_position(logits.data(),
                            static_cast<uint8_t>(current_player_index),
                            static_cast<uint16_t>(current_game_moves_));
        }

        // Update string move history
        move_history += std::to_string(current_game_moves_ + 1) + "." 
           + char('0' + (4 - chosen_move.x)) 
           + char('a' + chosen_move.y) 
           + " ";


        board.make_move(chosen_move, current_player);
        switch_player();
        current_game_moves_++;
    }

    Cell_state winner = board.check_winner();
    
    if (!is_evaluation) {
        pool_.finalize_game(winner);
    }

    // board.display_board(std::cout);
    return winner;
}

void Game::switch_player() {
    current_player_index = 1 - current_player_index;
}