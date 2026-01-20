#include "player.h"

#include <iostream>
#include <stdexcept>

#include "mcts_agent.h"
#include "minimax_agent.h"
#include <torch/torch.h>

std::pair<std::array<int, 4>, torch::Tensor> Human_player::choose_move(const Board& board,
                                              Cell_state player) {
  int move_index;
  bool valid_choice = false;
  torch::Tensor dummy_tensor = torch::zeros({1}, torch::kFloat32);
  std::vector<std::array<int, 4>> all_moves = board.get_valid_moves(player);
  board.print_valid_moves(all_moves);

  
  while (!valid_choice) {
      std::cout << "\n Choose one move among the given above: ";
      if (!(std::cin >> move_index)) {
          std::cout << "Invalid input! Try again." << std::endl;
          std::cin.clear();  // clear error flags
          std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
              '\n');  // ignore the rest of the line
          continue;
      }
      
      else {
          if (move_index < 1 || move_index > all_moves.size()) {
              std::cout << "Invalid choice! Try again." << std::endl;
              continue;
          }
          else {
              return {all_moves[move_index - 1], dummy_tensor};
          }
      }
  }

  return {{ -1, -1, -1, -1 }, dummy_tensor};  // should never reach this
}


Mcts_player::Mcts_player(double exploration_factor,
                         int number_iteration,
                         LogLevel log_level,
                         float temperature,
                         float dirichlet_alpha,
                         float dirichlet_epsilon,
                         std::shared_ptr<AlphaZModel> network,
                         int max_depth,
                         bool tree_reuse)
    : exploration_factor(exploration_factor),
      number_iteration(number_iteration),
      log_level(log_level),
      temperature(temperature),
      dirichlet_alpha(dirichlet_alpha),
      dirichlet_epsilon(dirichlet_epsilon),
      network(network),
      max_depth(max_depth),
      tree_reuse(tree_reuse) {
  
  // Parameter validation
  if (exploration_factor < 0.0) {
    throw std::invalid_argument("exploration_factor must be non-negative");
  }
  
  if (number_iteration <= 0) {
    throw std::invalid_argument("number_iteration must be positive");
  }
  
  if (temperature < 0.0f) {
    throw std::invalid_argument("temperature must be non-negative");
  }
  
  if (dirichlet_alpha <= 0.0f) {
    throw std::invalid_argument("dirichlet_alpha must be positive");
  }
  
  if (dirichlet_epsilon < 0.0f || dirichlet_epsilon > 1.0f) {
    throw std::invalid_argument("dirichlet_epsilon must be in range [0, 1]");
  }
}

std::pair<std::array<int, 4>, torch::Tensor> Mcts_player::choose_move(const Board& board,
                                             Cell_state player) {

    Mcts_agent agent(exploration_factor, 
                   number_iteration, 
                   log_level,
                   temperature,
                   dirichlet_alpha,
                   dirichlet_epsilon,
                   network,
                   max_depth,
                   tree_reuse);

  return agent.choose_move(board, player);
}

LogLevel Mcts_player::get_verbose_level() const { return log_level; }

void Mcts_player::set_temperature(float temp) { 
  if (temp < 0.0f) {
    throw std::invalid_argument("temperature must be non-negative");
  }
  temperature = temp; 
}

float Mcts_player::get_temperature() const { 
  return temperature; 
}

void Mcts_player::set_exploration_factor(double factor) {
  if (factor < 0.0) {
    throw std::invalid_argument("exploration_factor must be non-negative");
  }
  exploration_factor = factor;
}

double Mcts_player::get_exploration_factor() const {
  return exploration_factor;
}

void Mcts_player::set_number_iteration(int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("number_iteration must be positive");
  }
  number_iteration = iterations;
}

int Mcts_player::get_number_iteration() const {
  return number_iteration;
}

void Mcts_player::set_dirichlet_alpha(float alpha) {
  if (alpha <= 0.0f) {
    throw std::invalid_argument("dirichlet_alpha must be positive");
  }
  dirichlet_alpha = alpha;
}

float Mcts_player::get_dirichlet_alpha() const {
  return dirichlet_alpha;
}

void Mcts_player::set_dirichlet_epsilon(float epsilon) {
  if (epsilon < 0.0f || epsilon > 1.0f) {
    throw std::invalid_argument("dirichlet_epsilon must be in range [0, 1]");
  }
  dirichlet_epsilon = epsilon;
}

float Mcts_player::get_dirichlet_epsilon() const {
  return dirichlet_epsilon;
}

Minimax_player::Minimax_player(int max_depth,
                               bool use_alpha_beta,
                               LogLevel log_level)
    : max_depth(max_depth),
      use_alpha_beta(use_alpha_beta),
      log_level(log_level) {
  
  // Parameter validation
  if (max_depth <= 0) {
    throw std::invalid_argument("max_depth must be positive");
  }
}

std::pair<std::array<int, 4>, torch::Tensor> Minimax_player::choose_move(const Board& board,
                                                                          Cell_state player) {
  Minimax_agent agent(max_depth, 
                      use_alpha_beta, 
                      log_level);
  return agent.choose_move(board, player);
}

LogLevel Minimax_player::get_verbose_level() const {
  return log_level;
}

void Minimax_player::set_max_depth(int depth) {
  if (depth <= 0) {
    throw std::invalid_argument("max_depth must be positive");
  }
  max_depth = depth;
}

int Minimax_player::get_max_depth() const {
  return max_depth;
}