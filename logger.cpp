#include "logger.h"

// Initialize static member
std::shared_ptr<Logger> Logger::logger = nullptr;

std::shared_ptr<Logger> Logger::instance(LogLevel level) {
    if (!logger) {
        logger = std::shared_ptr<Logger>(new Logger(level));
    }
    return logger;
}

void Logger::log(const std::string& message) {
    if (log_level != LogLevel::NONE) {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << message << std::endl;
    }
}

bool Logger::should_log(LogLevel required_level) const {
    return static_cast<int>(log_level) >= static_cast<int>(required_level);
}

void Logger::log_mcts_start(Cell_state player) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "\n=============MCTS START - " << player
                << " to move=============\n";
        log(message.str());
    }
}

void Logger::log_mcts_end() {
    if (should_log(LogLevel::STEPS_ONLY)) {
        log("\n=============MCTS END=============\n");
    }
}

void Logger::log_iteration_number(int iteration_number) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "\n--- ITERATION " << iteration_number << " ---\n";
        log(message.str());
    }
}

void Logger::log_step(const std::string& step_name, const std::array<int, 4>& move) {
    if (should_log(LogLevel::STEPS_ONLY)) {
      std::ostringstream message;
      if (move[0]<0){
        message << "[" << step_name << "] Node: ROOT";
      }
      else{
        message << "[" << step_name << "] Node: " << print_move(move);
        
      }
      log(message.str());
    }
}

void Logger::log_nn_evaluation(const std::array<int, 4>& move, float value_from_nn, int num_legal_moves) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "[EVALUATION]" <<" Value from NN=" << std::fixed << std::setprecision(2) 
                << value_from_nn << ", Number of Legal Moves=" << num_legal_moves;
        log(message.str());
    }
}

void Logger::log_expansion(const std::array<int, 4>& move, int num_children) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "  Inititalized " << print_move(move) << " with " 
                << num_children << " children";
        log(message.str());
    }
}

void Logger::log_selected_child(const std::array<int, 4>& move, double puct_score) {
    if (should_log(LogLevel::SELECTION_ONLY)) {
        std::ostringstream message;
        message << "  Selected: " << print_move(move) << " | PUCT=";
        if (puct_score == std::numeric_limits<double>::max()) {
            message << "inf";
        } else {
            message << std::fixed << std::setprecision(4) << puct_score;
        }
        log(message.str());
    }
}

void Logger::log_puct_details(const std::array<int, 4>& move, float q_value, 
                              float u_value, float prior, int visits, int parent_visits) 
{
    // Clamp NaN values
    if (std::isnan(q_value)) q_value = 0.0f;
    if (std::isnan(u_value)) u_value = 0.0f;

    if (should_log(LogLevel::EVERYTHING)) {
        std::ostringstream message;
        message << "    " << print_move(move) << ": Q=" << std::fixed << std::setprecision(2) << q_value
                << ", U=" << std::setprecision(2) << u_value
                << ", P=" << std::setprecision(2) << prior
                << ", N=" << visits << "/" << parent_visits;
        log(message.str());
    }
}

void Logger::log_simulation_start(const std::array<int, 4>& move, const Board& board) {
    if (should_log(LogLevel::EVERYTHING)) {
        std::ostringstream message;
        std::ostringstream board_string;
        board.display_board(board_string);
        message << "\n  Random playout from " << print_move(move) 
                << ":\n" << board_string.str();
        log(message.str());
    }
}

void Logger::log_simulation_step(Cell_state current_player, const Board& board,
                                 const std::array<int, 4>& move) {
    if (should_log(LogLevel::EVERYTHING)) {
        std::ostringstream message;
        message << "    " << current_player << " plays " << print_move(move);
        log(message.str());
    }
}

void Logger::log_simulation_end(float value) {
    if (should_log(LogLevel::EVERYTHING)) {
        std::ostringstream message;
        message << "  Playout result: " << std::fixed << std::setprecision(2) << value;
        log(message.str());
    }
}

void Logger::log_backpropagation_start(const std::array<int, 4>& move, float value) {
    if (should_log(LogLevel::BACKPROP_ONLY)) {
        std::ostringstream message;
        message << "  Backprop value=" << std::fixed << std::setprecision(2) 
                << value << " from " << print_move(move);
        log(message.str());
    }
}

void Logger::log_backpropagation_result(const std::array<int, 4>& move,
                                       float acc_value, int visit_count) {
    if (should_log(LogLevel::BACKPROP_ONLY)) {
        std::ostringstream message;
        message << "    " << print_move(move) << ": Visits=" << visit_count
                << ",| Acc Value=" << std::fixed << std::setprecision(2) << acc_value
                << ",| Mean Value=" << std::setprecision(2) << acc_value/visit_count;
        log(message.str());
    }
}

void Logger::log_root_stats(int visit_count, size_t child_nodes) {
    if (should_log(LogLevel::ROOT_STATS)) {
        std::ostringstream message;
        message << "\n--- ROOT STATISTICS ---\n"
                << "Total visits: " << visit_count
                << " | Children: " << child_nodes << "\n";
        log(message.str());
    }
}

void Logger::log_child_node_stats(const std::array<int, 4>& move,
                                  float acc_value, int visit_count, 
                                  float prior_proba) {
    if (should_log(LogLevel::ROOT_STATS)) {
        std::ostringstream message;
        message << "  " << print_move(move)
                << " | Visits: " << visit_count
                << " | Prior: " << std::fixed << std::setprecision(2) << prior_proba
                << " | Mean Value: " << std::setprecision(2) << acc_value/visit_count
                << " | Acc Value: " << std::setprecision(2) << acc_value;
        log(message.str());
    }
}

void Logger::log_timer_ran_out(int iteration_counter) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "\n--- Completed " << iteration_counter 
                << " iterations. Selecting best move ---\n";
        log(message.str());
    }
}

void Logger::log_best_child_chosen(int iteration_counter,
                                   const std::array<int, 4>& move,
                                   float avg_value, int visits) {
    if (should_log(LogLevel::STEPS_ONLY)) {
        std::ostringstream message;
        message << "\n>>> FINAL CHOICE: " << print_move(move)
                << " | Visits: " << visits
                << " | Avg Value: " << std::fixed << std::setprecision(2) << avg_value
                << " | After " << iteration_counter << " iterations\n";
        log(message.str());
    }
}

void Logger::log_dirichlet_noise_applied(float alpha, float exploration_fraction) {
    if (should_log(LogLevel::EVERYTHING)) {
        std::ostringstream message;
        message << "  Applied Dirichlet noise: alpha=" << std::fixed 
                << std::setprecision(2) << alpha
                << ", exploration_fraction=" << std::setprecision(2) 
                << exploration_fraction;
        log(message.str());
    }
}

std::string Logger::print_move(std::array<int, 4> move) {
    char column = 'A' + move[1];  
    int row = move[0] + 1; 

    std::ostringstream oss;
    oss << "(" << row << column << ")";

    return oss.str();
}