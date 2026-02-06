#ifndef PLAYER_H
#define PLAYER_H

#include <chrono>
#include <utility>
#include <vector>
#include "logger.h"
#include "inference_queue_shm.h"
#include "board.h"

/**
 * @brief The Player class is an abstract base class for all game player types
 *
 * A Player's primary responsibility is to choose a move based on the current
 * state of the game board. This interaction is modelled via function choose_move().
 */
class Player {
 public:
  /**
   * @brief Abstract function for choosing a move on the game board
   *
   * @param board Current state of the board
   * @param player The cell of the current player
   *
   * @return The chosen move as an array of integers and policy tensor for data collection
   */
  virtual std::pair<Move, std::vector<float>> choose_move(
      const Board& board,
      Cell_state player) = 0;
  
  virtual ~Player() = default;
};

/**
 * @brief The Human_player class is a class that inherits from Player,
 * implementing the choose_move() function to allow user input for move selection,
 * with validation to ensure the move is legitimate
 */
class Human_player : public Player {
 public:
  /**
   * @brief Implementation of the choose_move function for the Human_player class
   *
   * This function prompts the user to make a choice of which move to make.
   * if the user choose a move that is not in the given list, this function prompts the user to try again.
   *
   * @param board Current state of the board
   * @param player The cell of the current player
   *
   * @return The chosen move as an array of integers and a dummy policy tensor (we don't get data from human games)
   */
  std::pair<Move, std::vector<float>> choose_move(
      const Board& board,
      Cell_state player) override;
};


class Mcts_player_selfplay : public Player {
 public:
  /**
   * @brief Mcts_player_selfplay constructor initializes MCTS parameters for self-play
   *
   * @param exploration_factor Exploration factor (PUCT constant)
   * @param number_iteration Iteration number for MCTS simulations
   * @param log_level Log Level for debugging output
   * @param temperature Temperature parameter for action selection 
   * @param dirichlet_alpha Alpha parameter for Dirichlet noise 
   * @param dirichlet_epsilon Epsilon for mixing Dirichlet noise with policy
   * @param inference_queue Pointer to shared inference queue for batched evaluation
   * @param max_depth Maximum search depth for MCTS
   * @param tree_reuse Whether to reuse the search tree between moves
   */
  Mcts_player_selfplay(double exploration_factor,
                       int number_iteration,
                       LogLevel log_level = LogLevel::NONE,
                       float temperature = 0.0f,
                       float dirichlet_alpha = 0.3f,
                       float dirichlet_epsilon = 0.25f,
                       std::shared_ptr<SharedMemoryInferenceQueue> queue = nullptr,
                       int max_depth = 100,
                       bool tree_reuse = false,
                       uint32_t model_id = 0);
  
  /**
   * @brief Implementation of the choose_move function for the Mcts_player_selfplay class
   *
   * @param board The current state of the game board
   * @param player The current player 
   *
   * @return The chosen move as an array of integers and policy tensor for data collection
   */
  std::pair<Move, std::vector<float>> choose_move(
      const Board& board,
      Cell_state player) override;
  
  /**
   * @brief Getter for verbose level private member of the Mcts_player_selfplay class
   *
   * @return The verbose level
   */
  LogLevel get_verbose_level() const;
  
  void set_temperature(float temp);
  float get_temperature() const;
  
  void set_exploration_factor(double factor);
  double get_exploration_factor() const;
  
  void set_number_iteration(int iterations);
  int get_number_iteration() const;
  
  void set_dirichlet_alpha(float alpha);
  float get_dirichlet_alpha() const;
  
  void set_dirichlet_epsilon(float epsilon);
  float get_dirichlet_epsilon() const;
  
 private:
  double exploration_factor;
  int number_iteration;
  LogLevel log_level;
  float temperature;
  float dirichlet_alpha;
  float dirichlet_epsilon;
  std::shared_ptr<SharedMemoryInferenceQueue> queue;
  int max_depth;
  bool tree_reuse;
  uint32_t model_id;
};

#endif