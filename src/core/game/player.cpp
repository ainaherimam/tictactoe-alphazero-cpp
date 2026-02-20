#include "core/game/player.h"
#include "core/game/board.h"
#include <iostream>
#include "core/mcts/mcts_agent_selfplay.h"
#include "core/mcts/mcts_agent_triton.h"
#include "inference/shared_memory/inference_queue_shm.h"
#include <vector>

std::pair<Move, std::vector<float>> Human_player::choose_move(const Board& board,
                                              Cell_state player) {
  int move_index;
  bool valid_choice = false;
  std::vector<float> dummy(16, 0.0f);
  std::vector<Move> all_moves = board.get_valid_moves(player);
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
              return {all_moves[move_index - 1], dummy};
          }
      }
  }

  return {{ -1, -1, -1, -1 }, dummy};  // should never reach this
}


Mcts_player_selfplay::Mcts_player_selfplay(const Mcts_config& config)
    : exploration_factor(config.exploration_factor),
      number_iteration(config.number_iteration),
      log_level(config.log_level),
      temperature(config.temperature),
      dirichlet_alpha(config.dirichlet_alpha),
      dirichlet_epsilon(config.dirichlet_epsilon),
      queue(config.queue),
      max_depth(config.max_depth),
      tree_reuse(config.tree_reuse),
      model_id(config.model_id) {

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

std::pair<Move, std::vector<float>> Mcts_player_selfplay::choose_move(
    const Board& board,
    Cell_state player) {
    Mcts_config config(exploration_factor, 
                      number_iteration, 
                      log_level,
                      temperature,
                      dirichlet_alpha,
                      dirichlet_epsilon,
                      queue,
                      max_depth,
                      tree_reuse,
                      model_id);
    Mcts_agent_selfplay agent(config);
    return agent.choose_move(board, player);
}

LogLevel Mcts_player_selfplay::get_verbose_level() const { 
  return log_level; 
}

void Mcts_player_selfplay::set_temperature(float temp) { 
  if (temp < 0.0f) {
    throw std::invalid_argument("temperature must be non-negative");
  }
  temperature = temp; 
}

float Mcts_player_selfplay::get_temperature() const { 
  return temperature; 
}

void Mcts_player_selfplay::set_exploration_factor(double factor) {
  if (factor < 0.0) {
    throw std::invalid_argument("exploration_factor must be non-negative");
  }
  exploration_factor = factor;
}

double Mcts_player_selfplay::get_exploration_factor() const {
  return exploration_factor;
}

void Mcts_player_selfplay::set_number_iteration(int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("number_iteration must be positive");
  }
  number_iteration = iterations;
}

int Mcts_player_selfplay::get_number_iteration() const {
  return number_iteration;
}

void Mcts_player_selfplay::set_dirichlet_alpha(float alpha) {
  if (alpha <= 0.0f) {
    throw std::invalid_argument("dirichlet_alpha must be positive");
  }
  dirichlet_alpha = alpha;
}

float Mcts_player_selfplay::get_dirichlet_alpha() const {
  return dirichlet_alpha;
}

void Mcts_player_selfplay::set_dirichlet_epsilon(float epsilon) {
  if (epsilon < 0.0f || epsilon > 1.0f) {
    throw std::invalid_argument("dirichlet_epsilon must be in range [0, 1]");
  }
  dirichlet_epsilon = epsilon;
}

float Mcts_player_selfplay::get_dirichlet_epsilon() const {
  return dirichlet_epsilon;
}



Mcts_player_triton::Mcts_player_triton(const Mcts_triton_config& config)
    : exploration_factor(config.exploration_factor),
      number_iteration(config.number_iteration),
      log_level(config.log_level),
      temperature(config.temperature),
      dirichlet_alpha(config.dirichlet_alpha),
      dirichlet_epsilon(config.dirichlet_epsilon),
      client(config.client),
      max_depth(config.max_depth),
      tree_reuse(config.tree_reuse),
      model_id(config.model_id) {

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

std::pair<Move, std::vector<float>> Mcts_player_triton::choose_move(
    const Board& board,
    Cell_state player) {
    Mcts_triton_config config(exploration_factor, 
                             number_iteration, 
                             log_level,
                             temperature,
                             dirichlet_alpha,
                             dirichlet_epsilon,
                             client,
                             max_depth,
                             tree_reuse,
                             model_id);
    Mcts_agent_triton agent(config);
    return agent.choose_move(board, player);
}

LogLevel Mcts_player_triton::get_verbose_level() const { 
  return log_level; 
}

void Mcts_player_triton::set_temperature(float temp) { 
  if (temp < 0.0f) {
    throw std::invalid_argument("temperature must be non-negative");
  }
  temperature = temp; 
}

float Mcts_player_triton::get_temperature() const { 
  return temperature; 
}

void Mcts_player_triton::set_exploration_factor(double factor) {
  if (factor < 0.0) {
    throw std::invalid_argument("exploration_factor must be non-negative");
  }
  exploration_factor = factor;
}

double Mcts_player_triton::get_exploration_factor() const {
  return exploration_factor;
}

void Mcts_player_triton::set_number_iteration(int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("number_iteration must be positive");
  }
  number_iteration = iterations;
}

int Mcts_player_triton::get_number_iteration() const {
  return number_iteration;
}

void Mcts_player_triton::set_dirichlet_alpha(float alpha) {
  if (alpha <= 0.0f) {
    throw std::invalid_argument("dirichlet_alpha must be positive");
  }
  dirichlet_alpha = alpha;
}

float Mcts_player_triton::get_dirichlet_alpha() const {
  return dirichlet_alpha;
}

void Mcts_player_triton::set_dirichlet_epsilon(float epsilon) {
  if (epsilon < 0.0f || epsilon > 1.0f) {
    throw std::invalid_argument("dirichlet_epsilon must be in range [0, 1]");
  }
  dirichlet_epsilon = epsilon;
}

float Mcts_player_triton::get_dirichlet_epsilon() const {
  return dirichlet_epsilon;
}