#include "player.h"

#include <chrono>
#include <iostream>

#include "mcts_agent.h"
#include <torch/torch.h>

std::pair<std::array<int, 4>,torch::Tensor> Human_player::choose_move(const Board& board,
                                              Cell_state player) {
  int choice;
  bool valid_choice = false;
  torch::Tensor dummy_tensor = torch::zeros({1}, torch::kFloat32);
  std::vector<std::array<int, 4>> all_moves = board.get_valid_moves(player);
  board.print_valid_moves(all_moves);

  
  while (!valid_choice) {
      std::cout << "\n Choose one move among the given above: ";
      if (!(std::cin >> choice)) {
          std::cout << "Invalid input! Try again." << std::endl;
          std::cin.clear();  // clear error flags
          std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
              '\n');  // ignore the rest of the line
          continue;
      }
      
      else {
          if (choice < 1 || choice > all_moves.size()) {
              std::cout << "Invalid choice! Try again." << std::endl;
              continue;
          }
          else {
              return {all_moves[choice-1], dummy_tensor};
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
                         torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network,
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
      tree_reuse(tree_reuse) {}

std::pair<std::array<int, 4>,torch::Tensor> Mcts_player::choose_move(const Board& board,
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

void Mcts_player::set_temperature(float temp) { temperature = temp; }


PolicyNetwork_player::PolicyNetwork_player(
    torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> network,
    float temperature,
    LogLevel log_level)
    : network(network),
      temperature(temperature),
      log_level(log_level) {}

std::pair<std::array<int, 4>, torch::Tensor> PolicyNetwork_player::choose_move(
    const Board& board,
    Cell_state player) {
  
    if (!network) {
    throw std::runtime_error("PolicyNetwork_player: Network is not initialized!");
    }

    Mcts_agent agent(1.5, 
                100, 
                log_level,
                temperature,
                0.3,
                0.25,
                network,
                -1,
                false);

    return agent.choose_move(board, player);

}

LogLevel PolicyNetwork_player::get_verbose_level() const { 
  return log_level; 
}