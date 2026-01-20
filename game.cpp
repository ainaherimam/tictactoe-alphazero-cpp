#include "game.h"
#include "logger.h"
#include <torch/torch.h>
#include <chrono>
#include "alphaz_model.h"

#include <iostream>
#include <random>


static std::mt19937 random_generator(std::random_device{}());


Game::Game(int board_size, std::unique_ptr<Player> player1,
           std::unique_ptr<Player> player_2, GameDataset& dataset, bool evaluation)
    : board(board_size), current_player_index(0), dataset_(dataset), evaluation(evaluation) {
  players[0] = std::move(player1);
  players[1] = std::move(player_2);
}

Cell_state Game::play() {

    int move_counter = 0;
    int max_move = 70;

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
            auto* mcts = dynamic_cast<Mcts_player*>(players[current_player_index].get());
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


        int chosen_x = chosen_move[0];
        int chosen_y = chosen_move[1];
        int chosen_dir = chosen_move[2];
        int chosen_tar = chosen_move[3];

        board.make_move(chosen_x, chosen_y, chosen_dir, chosen_tar, current_player);
        switch_player();

        move_counter ++;
    }

    Cell_state winner = board.check_winner();
    //update the z target on the data based on the winner
    if (!evaluation){
        dataset_.update_last_z(result_z, winner);
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

Cell_state Game::evaluate_play() {

    int move_counter = 0;;

    Cell_state current_player =
            current_player_index == 0 ? Cell_state::X : Cell_state::O;

    while (board.check_winner() == Cell_state::Empty) {

        auto valid_moves = board.get_valid_moves(current_player);
        if (valid_moves.empty()) {
            break;
        }

        Cell_state current_player = current_player_index == 0 ? Cell_state::X : Cell_state::O;
        auto [chosen_move, logits] = players[current_player_index]->choose_move(board, current_player);

        int chosen_x = chosen_move[0];
        int chosen_y = chosen_move[1];
        int chosen_dir = chosen_move[2];
        int chosen_tar = chosen_move[3];
        
        board.make_move(chosen_x, chosen_y, chosen_dir, chosen_tar, current_player);

        switch_player();
        move_counter ++;
    }
    Cell_state winner = board.check_winner();
    return winner;
}


std::string Game::print_move(std::array<int, 4> move) {

    // Print the row as a number and the column as an alphabet
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

