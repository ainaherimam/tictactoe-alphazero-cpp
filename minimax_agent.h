#ifndef MINIMAX_AGENT_H
#define MINIMAX_AGENT_H

#include <array>
#include <utility>

#include "board.h"
#include "logger.h"
#include <torch/torch.h>

/**
 * @brief The Minimax_agent class implements the minimax algorithm for game playing
 *
 * This agent uses adversarial search to find the optimal move by recursively
 * evaluating the game tree. It supports alpha-beta pruning for optimization.
 */
class Minimax_agent {
 public:
  /**
   * @brief Constructor for Minimax_agent
   *
   * @param max_depth Maximum search depth for the minimax algorithm
   * @param use_alpha_beta Whether to use alpha-beta pruning
   * @param log_level Log level for debugging output
   */
  Minimax_agent(int max_depth,
                bool use_alpha_beta = true,
                LogLevel log_level = LogLevel::NONE);

  /**
   * @brief Choose the best move using minimax algorithm
   *
   * @param board The current state of the game board
   * @param player The current player
   *
   * @return The chosen move as an array of integers and a dummy policy tensor
   */
  std::pair<std::array<int, 4>, torch::Tensor> choose_move(const Board& board,
                                                             Cell_state player);

 private:
  /**
   * @brief Minimax algorithm with alpha-beta pruning
   *
   * @param board The current state of the game board
   * @param depth Current search depth
   * @param alpha Alpha value for pruning
   * @param beta Beta value for pruning
   * @param is_maximizing Whether current player is maximizing
   * @param player The current player
   *
   * @return The evaluation score of the position
   */
  int minimax(const Board& board,
              int depth,
              int alpha,
              int beta,
              bool is_maximizing,
              Cell_state player);

  /**
   * @brief Minimax algorithm without alpha-beta pruning
   *
   * @param board The current state of the game board
   * @param depth Current search depth
   * @param is_maximizing Whether current player is maximizing
   * @param player The current player
   *
   * @return The evaluation score of the position
   */
  int minimax_no_pruning(const Board& board,
                         int depth,
                         bool is_maximizing,
                         Cell_state player);

  /**
   * @brief Evaluate the board position for the given player
   *
   * @param board The current state of the game board
   * @param player The player to evaluate for
   *
   * @return The evaluation score
   */
  int evaluate(const Board& board, Cell_state player);

  /**
   * @brief Get the opponent of the given player
   *
   * @param player The current player
   *
   * @return The opponent player
   */
  Cell_state get_opponent(Cell_state player) const;

  int max_depth;
  bool use_alpha_beta;
  LogLevel log_level;
  int nodes_evaluated;
};

#endif