#include "minimax_agent.h"
#include <iostream>
#include <limits>

Minimax_agent::Minimax_agent(int max_depth,
                             bool use_alpha_beta,
                             LogLevel log_level)
    : max_depth(max_depth),
      use_alpha_beta(use_alpha_beta),
      log_level(log_level),
      nodes_evaluated(0) {}

std::pair<std::array<int, 4>, torch::Tensor> Minimax_agent::choose_move(
    const Board& board,
    Cell_state player) {
  
  nodes_evaluated = 0;
  
  std::vector<std::array<int, 4>> valid_moves = board.get_valid_moves(player);
  
  if (valid_moves.empty()) {
    if (log_level > LogLevel::NONE) {
      std::cout << "No valid moves available!" << std::endl;
    }
    torch::Tensor dummy_tensor = torch::zeros({1}, torch::kFloat32);
    return {{-1, -1, -1, -1}, dummy_tensor};
  }

  std::array<int, 4> best_move = valid_moves[0];
  int best_score = std::numeric_limits<int>::min();

  if (log_level > LogLevel::NONE) {
    std::cout << "Evaluating " << valid_moves.size() << " possible moves..." << std::endl;
  }

  for (const auto& move : valid_moves) {
    Board temp_board = board;
    temp_board.make_move(move[0], move[1], move[2], move[3], player);
    
    int score;
    if (use_alpha_beta) {
      score = minimax(temp_board, 1, std::numeric_limits<int>::min(), 
                     std::numeric_limits<int>::max(), false, player);
    } else {
      score = minimax_no_pruning(temp_board, 1, false, player);
    }

    if (log_level > LogLevel::NONE) {
      std::cout << "Move (" << move[0] << "," << move[1] << ") -> Score: " << score << std::endl;
    }

    if (score > best_score) {
      best_score = score;
      best_move = move;
    }
  }

  if (log_level >= LogLevel::STEPS_ONLY) {
    std::cout << "Best move: (" << best_move[0] << "," << best_move[1] 
              << ") with score " << best_score << std::endl;
    std::cout << "Nodes evaluated: " << nodes_evaluated << std::endl;
  }

  // Return move with dummy policy tensor
  torch::Tensor dummy_tensor = torch::zeros({1}, torch::kFloat32);
  return {best_move, dummy_tensor};
}

int Minimax_agent::minimax(const Board& board,
                           int depth,
                           int alpha,
                           int beta,
                           bool is_maximizing,
                           Cell_state player) {
  
  nodes_evaluated++;

  Cell_state winner = board.check_winner();
  if (winner != Cell_state::Empty) {
    if (winner == player) {
      return 1000 - depth; 
    } else {
      return -1000 + depth;
    }
  }


  if (depth >= max_depth) {
    return evaluate(board, player);
  }

  Cell_state current_player = is_maximizing ? player : get_opponent(player);
  std::vector<std::array<int, 4>> valid_moves = board.get_valid_moves(current_player);

  if (valid_moves.empty()) {
    return 0; 
  }

  if (is_maximizing) {
    int max_eval = std::numeric_limits<int>::min();
    
    for (const auto& move : valid_moves) {
      Board temp_board = board;
      temp_board.make_move(move[0], move[1], move[2], move[3], current_player);
      
      int eval = minimax(temp_board, depth + 1, alpha, beta, false, player);
      max_eval = std::max(max_eval, eval);
      
      alpha = std::max(alpha, eval);
      if (beta <= alpha) {
        break;  // Beta cutoff
      }
    }
    
    return max_eval;
  } else {
    int min_eval = std::numeric_limits<int>::max();
    
    for (const auto& move : valid_moves) {
      Board temp_board = board;
      temp_board.make_move(move[0], move[1], move[2], move[3], current_player);
      
      int eval = minimax(temp_board, depth + 1, alpha, beta, true, player);
      min_eval = std::min(min_eval, eval);
      
      beta = std::min(beta, eval);
      if (beta <= alpha) {
        break;  // Alpha cutoff
      }
    }
    
    return min_eval;
  }
}

int Minimax_agent::minimax_no_pruning(const Board& board,
                                      int depth,
                                      bool is_maximizing,
                                      Cell_state player) {
  
  nodes_evaluated++;

  // Check terminal conditions
  Cell_state winner = board.check_winner();
  if (winner != Cell_state::Empty) {
    if (winner == player) {
      return 1000 - depth;
    } else {
      return -1000 + depth;
    }
  }

  // Check depth limit
  if (depth >= max_depth) {
    return evaluate(board, player);
  }

  Cell_state current_player = is_maximizing ? player : get_opponent(player);
  std::vector<std::array<int, 4>> valid_moves = board.get_valid_moves(current_player);

  // No valid moves available - it's a draw
  if (valid_moves.empty()) {
    return 0;
  }

  if (is_maximizing) {
    int max_eval = std::numeric_limits<int>::min();
    
    for (const auto& move : valid_moves) {
      Board temp_board = board;
      temp_board.make_move(move[0], move[1], move[2], move[3], current_player);
      
      int eval = minimax_no_pruning(temp_board, depth + 1, false, player);
      max_eval = std::max(max_eval, eval);
    }
    
    return max_eval;
  } else {
    int min_eval = std::numeric_limits<int>::max();
    
    for (const auto& move : valid_moves) {
      Board temp_board = board;
      temp_board.make_move(move[0], move[1], move[2], move[3], current_player);
      
      int eval = minimax_no_pruning(temp_board, depth + 1, true, player);
      min_eval = std::min(min_eval, eval);
    }
    
    return min_eval;
  }
}

int Minimax_agent::evaluate(const Board& board, Cell_state player) {
  Cell_state opponent = get_opponent(player);
  
  int score = 0;
  
  // Heuristic evaluation based on mobility
  // More available moves generally means better position
  int player_mobility = board.get_valid_moves(player).size();
  int opponent_mobility = board.get_valid_moves(opponent).size();
  
  score += (player_mobility - opponent_mobility) * 5;
  
  return score;
}

Cell_state Minimax_agent::get_opponent(Cell_state player) const {
  if (player == Cell_state::X) {
    return Cell_state::O;
  } else if (player == Cell_state::O) {
    return Cell_state::X;
  }
  return Cell_state::Empty;
}