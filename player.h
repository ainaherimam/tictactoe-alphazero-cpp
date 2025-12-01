#ifndef PLAYER_H
#define PLAYER_H

#include <chrono>
#include <utility>

#include "board.h"
#include "logger.h"
#include "nn_model.h"
#include <torch/torch.h>

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
  virtual std::pair<std::array<int, 4>, torch::Tensor> choose_move(
      const Board& board,
      Cell_state player) = 0;
};

/**
 * @brief The Human_player class is a concrete class that inherits from Player,
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
  std::pair<std::array<int, 4>, torch::Tensor> choose_move(
      const Board& board,
      Cell_state player) override;
};

class Mcts_player : public Player {
 public:
  /**
   * @brief Mcts_player constructor initializes MCTS parameters
   *
   * @param exploration_factor Exploration factor for MCTS (PUCT constant)
   * @param number_iteration Maximum iteration number for MCTS simulations
   * @param log_level Log Level for debugging output
   * @param temperature Temperature parameter for action selection 
   * @param dirichlet_alpha Alpha parameter for Dirichlet noise 
   * @param dirichlet_epsilon Epsilon for mixing Dirichlet noise with policy
   * @param network Neural network model for policy/value evaluation 
   * @param max_depth Maximum search depth for MCTS
   * @param tree_reuse Whether to reuse the search tree between moves
   */
  Mcts_player(double exploration_factor,
              int number_iteration,
              LogLevel log_level = LogLevel::NONE,
              float temperature = 0.0f,
              float dirichlet_alpha = 0.3f,
              float dirichlet_epsilon = 0.25f,
              torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network = nullptr,
              int max_depth = -1,
              bool tree_reuse = false);

  /**
   * @brief Implementation of the choose_move function for the Mcts_player class
   *
   * @param board The current state of the game board
   * @param player The current player 
   *
   * @return The chosen move as an array of integers and policy tensor for data collection
   */
  std::pair<std::array<int, 4>, torch::Tensor> choose_move(
      const Board& board,
      Cell_state player) override;

  /**
   * @brief Getter for verbose level private member of the Mcts_player class
   *
   * @return The verbose level
   */
  LogLevel get_verbose_level() const;
  void set_temperature(float temp);


 private:
  double exploration_factor;
  int number_iteration;
  LogLevel log_level;
  float temperature;
  float dirichlet_alpha;
  float dirichlet_epsilon;
  torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network;  
  int max_depth;
  bool tree_reuse;
};

/**
 * @brief PolicyNetwork_player uses a neural network's policy head directly
 * to choose moves without MCTS search
 *
 * This player evaluates the board position using a trained neural network
 * and selects moves based on the policy output, optionally with temperature
 * sampling for exploration.
 */
class PolicyNetwork_player : public Player {
 public:
  /**
   * @brief Constructor for PolicyNetwork_player
   *
   * @param network Neural network model for policy evaluation
   * @param temperature Temperature for action selection
   * @param log_level Log level for debugging output
   */
  PolicyNetwork_player(torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network,
                       float temperature = 0.0f,
                       LogLevel log_level = LogLevel::NONE);

  /**
   * @brief Implementation of choose_move using neural network policy
   *
   * @param board The current state of the game board
   * @param player The current player
   *
   * @return The chosen move as an array of integers and policy tensor
   */
  std::pair<std::array<int, 4>, torch::Tensor> choose_move(
      const Board& board,
      Cell_state player) override;

  /**
   * @brief Getter for verbose level
   *
   * @return The verbose level
   */
  LogLevel get_verbose_level() const;

 private:
  torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network;  // Neural network model
  float temperature;          // Temperature for move selection
  LogLevel log_level;         // Verbose level
};

#endif