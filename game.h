#ifndef GAME_H
#define GAME_H

#include <memory>

#include "board.h"
#include "cell_state.h"
#include "player.h"
#include "alphaz_model.h"

/**
 * @class Game
 * @brief Represents a Tic Tac Toe game.
 * 
 * This class defines a complete Tic Tac Toe game, including the game board and the
 * two players. It handles the game loop, player turns, and game state
 * transitions and also Data collection.
 */
class Game {
 public:
  /**
   * @brief Constructs a Game object.
   * 
   * Initializes a new game with a specified board size and two
   * players.
   * 
   * @param board_size The size of the game board.
   * @param player_1 Unique pointer to the first player.
   * @param player_2 Unique pointer to the second player.
   * @param dataset Reference to the game dataset for storing game data.
   * @param evaluation Flag for evaluation (default: false).
   */
  Game(int board_size, std::unique_ptr<Player> player_1,
       std::unique_ptr<Player> player_2,
       GameDataset& dataset, bool evaluation = false);

  /**
   * @brief Converts a move array to its string representation.
   * 
   * @param moves Array of 4 integers representing a move.
   * 
   * @return String representation of the move.
   */
  std::string print_move(std::array<int, 4> moves);

  /**
   * @brief Starts and manages the game loop for Self Play.
   * 
   * This function contains the main game loop for Self Play and Data collection (no logs). It continues until a player
   * wins.
   * 
   * @return The winning player's cell state. or '.' if dr
   */
  Cell_state play();

  /**
   * @brief Starts and manages  game loop for AI vs AI / Human vs Human / Human vs AI.
   * 
   * @return The winning player's cell state.
   */
  Cell_state simple_play();

  Cell_state evaluate_play();

  const Board& getBoard() const {
        return board;
    }

 private:
  Board board;
  std::unique_ptr<Player> players[2];
  int current_player_index;
  std::vector<torch::Tensor> result_z;
  bool evaluation = false;

  /**
   * @brief A cirular replay buffer
   */
  GameDataset& dataset_;

  /**
   * @brief Switches the current player.
   * 
   * This function switches the turn to the other player.
   */
  void switch_player();

  /**
   * @brief Executes a specified number of random moves on the current board.
   * 
   * @param random_move_number Number of random moves to perform.
   */
  void random_move(int random_move_number);
};

#endif