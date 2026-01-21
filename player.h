#ifndef PLAYER_H
#define PLAYER_H

#include <chrono>
#include <utility>

#include "board.h"
#include "logger.h"
#include "alphaz_model.h"
#include "mcts_agent_parallel.h"
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
  std::pair<std::array<int, 4>, torch::Tensor> choose_move(
      const Board& board,
      Cell_state player) override;
};

class Mcts_player : public Player {
 public:
  /**
   * @brief Mcts_player constructor initializes MCTS parameters
   *
   * @param exploration_factor Exploration factor (PUCT constant)
   * @param number_iteration Iteration number for MCTS simulations
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
              std::shared_ptr<AlphaZModel> network = nullptr,
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
  std::shared_ptr<AlphaZModel> network;  
  int max_depth;
  bool tree_reuse;
};

class Mcts_player_parallel : public Player {

 public:

  /**
   * @brief Mcts_player_parallel constructor initializes parallel MCTS parameters
   *
   * @param exploration_factor Exploration factor (PUCT constant)
   * @param number_iteration Iteration number for MCTS simulations
   * @param log_level Log Level for debugging output
   * @param temperature Temperature parameter for action selection 
   * @param dirichlet_alpha Alpha parameter for Dirichlet noise 
   * @param dirichlet_epsilon Epsilon for mixing Dirichlet noise with policy
   * @param network Neural network model for policy/value evaluation 
   * @param max_depth Maximum search depth for MCTS
   * @param tree_reuse Whether to reuse the search tree between moves
   * @param virtual_loss Virtual loss value for parallel search
   * @param num_workers Number of parallel worker threads
   * @param nn_batch_size Batch size for neural network evaluation
   */
  Mcts_player_parallel(double exploration_factor,
                       int number_iteration,
                       LogLevel log_level = LogLevel::NONE,
                       float temperature = 0.0f,
                       float dirichlet_alpha = 0.3f,
                       float dirichlet_epsilon = 0.25f,
                       std::shared_ptr<AlphaZModel> network = nullptr,
                       int max_depth = -1,
                       bool tree_reuse = false,
                       float virtual_loss = 1.0f,
                       int num_workers = 4,
                       int nn_batch_size = 8);

  /**
   * @brief Implementation of the choose_move function for the Mcts_player_parallel class
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
   * @brief Getter for verbose level private member of the Mcts_player_parallel class
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

  void set_virtual_loss(float vl);
  float get_virtual_loss() const;

  void set_num_workers(int workers);
  int get_num_workers() const;

  void set_nn_batch_size(int batch_size);
  int get_nn_batch_size() const;

 private:
  double exploration_factor;
  int number_iteration;
  LogLevel log_level;
  float temperature;
  float dirichlet_alpha;
  float dirichlet_epsilon;
  std::shared_ptr<AlphaZModel> network;
  int max_depth;
  bool tree_reuse;
  float virtual_loss;
  int num_workers;
  int nn_batch_size;
};

class Minimax_player : public Player {
 public:
  /**
   * @brief Constructor for Minimax_player
   *
   * @param max_depth Maximum search depth for minimax algorithm
   * @param use_alpha_beta Whether to use alpha-beta pruning optimization
   * @param log_level Log level for debugging output
   */
  Minimax_player(int max_depth,
                 bool use_alpha_beta = true,
                 LogLevel log_level = LogLevel::NONE);

  /**
   * @brief Implementation of choose_move using minimax algorithm
   *
   * @param board The current state of the game board
   * @param player The current player
   *
   * @return The chosen move as an array of integers and dummy policy tensor
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
  
  void set_max_depth(int depth);
  int get_max_depth() const;

 private:
  int max_depth;
  bool use_alpha_beta;
  LogLevel log_level;
};

#endif