#ifndef LOGGER_H
#define LOGGER_H

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <iomanip>
#include <limits>

#include "board.h"


/**
 * @brief Enumeration for different logging verbosity levels
 * 
 * Controls what information is logged during MCTS execution
 */
enum class LogLevel {
    NONE = 0,           
    STEPS_ONLY = 1,     // Only MCTS steps (selection, expansion, simulation, backprop)
    BACKPROP_ONLY = 2,
    SELECTION_ONLY = 3,
    ROOT_STATS = 4, 
    EVERYTHING = 5
};

/**
 * @brief Thread-safe singleton logger for MCTS debugging and analysis
 * 
 * Provides hierarchical logging capabilities for different stages of MCTS
 * including selection, expansion, simulation and backpropagation.
 */
class Logger {
private:
    LogLevel log_level;
    std::mutex mutex;
    static std::shared_ptr<Logger> logger;

    /**
     * @brief Private constructor for singleton pattern
     * 
     * @param level logging level
     */
    Logger(LogLevel level) : log_level(level) {}

    /**
     * @brief Internal logging method with thread safety
     * 
     * @param message The message to log
     */
    void log(const std::string& message);
    
    /**
     * @brief Checks if message should be logged at current level
     * 
     * @param required_level Minimum level required for logging
     * @return true if message should be logged, false otherwise
     */
    bool should_log(LogLevel required_level) const;

public:
    /**
     * @brief Get or create the Logger instance
     * 
     * @param level Logging level (only used on first call)
     * @return Shared pointer to Logger instance
     */
    static std::shared_ptr<Logger> instance(LogLevel level = LogLevel::NONE);
    
    /**
     * @brief Set the current logging level
     * 
     * @param level Logging level
     */
    void set_log_level(LogLevel level) { log_level = level; }
    
    /**
     * @brief Get the current logging level
     * 
     * @return Current LogLevel
     */
    LogLevel get_log_level() const { return log_level; }

    /**
     * @brief Format a move into move notation string
     * 
     * @param move Array containing move coordinates [from_x, from_y, to_x, to_y]
     * @return Formatted move string
     */
    std::string print_move(std::array<int, 4> move);

    // ========== MCTS Lifecycle Logging (Level 1 - STEPS_ONLY) ==========
    
    /**
     * @brief Log the start of MCTS search
     * 
     * @param player The player making the move
     */
    void log_mcts_start(Cell_state player);
    
    /**
     * @brief Log the end of MCTS search
     */
    void log_mcts_end();
    
    /**
     * @brief Log the current iteration number
     * 
     * @param iteration_number Current MCTS iteration
     */
    void log_iteration_number(int iteration_number);
    
    /**
     * @brief Log a specific MCTS step
     * 
     * @param step_name Name of the step (e.g., "Selection", "Expansion")
     * @param move Move
     */
    void log_step(const std::string& step_name, const std::array<int, 4>& move);
    
    // ========== Neural Network Logging (Level 1 - STEPS_ONLY) ==========
    
    /**
     * @brief Log neural network evaluation results
     * 
     * @param move Move that lead to the node to be evaluated
     * @param value_from_nn Value prediction from neural network
     * @param num_legal_moves Number of legal moves available
     */
    void log_nn_evaluation(const std::array<int, 4>& move, float value_from_nn, int num_legal_moves);
    
    /**
     * @brief Log node expansion details
     * 
     * @param move Move that lead to the node that is expanded
     * @param num_children Number of child nodes created
     */
    void log_expansion(const std::array<int, 4>& move, int num_children);
    
    // ========== Selection Logging (Level 3 - SELECTION_ONLY and above) ==========
    
    /**
     * @brief Log the selected child node
     * 
     * @param move Move that lead to selected child
     * @param puct_score PUCT score used for selection
     */
    void log_selected_child(const std::array<int, 4>& move, double puct_score);
    
    /**
     * @brief Log detailed PUCT calculation components
     * 
     * @param move Move being considered
     * @param q_value Mean value
     * @param u_value Exploration term
     * @param prior Prior probability from neural network
     * @param visits Visit count for this node
     * @param parent_visits Visit count for parent node
     */
    void log_puct_details(const std::array<int, 4>& move, float q_value, 
                          float u_value, float prior, int visits, int parent_visits);
    
    // ========== Simulation Logging (for Vanilla MCTS) ==========
    
    /**
     * @brief Log the start of a simulation and print the board (legacy method) 
     * 
     * @param move Starting move for simulation
     * @param board Current board state
     */
    void log_simulation_start(const std::array<int, 4>& move, const Board& board);
    
    /**
     * @brief Log a step during simulation for Vanilla MCTS(legacy method)
     * 
     * @param current_player Player making the move
     * @param board Current board state
     * @param move Move being made
     */
    void log_simulation_step(Cell_state current_player, const Board& board,
                             const std::array<int, 4>& move);
    
    /**
     * @brief Log the end of simulation (legacy method)
     * 
     * @param value Final value from simulation
     */
    void log_simulation_end(float value);
    
    // ========== Backpropagation Logging (Level 2 - BACKPROP_ONLY and above) ==========
    
    /**
     * @brief Log the start of backpropagation
     * 
     * @param move Move that lead to the node where backpropagation starts
     * @param value Value being backpropagated
     */
    void log_backpropagation_start(const std::array<int, 4>& move, float value);
    
    /**
     * @brief Log backpropagation update results
     * 
     * @param move Move
     * @param acc_value Accumulated value after update
     * @param visit_count Visit count after update
     */
    void log_backpropagation_result(const std::array<int, 4>& move,
                                    float acc_value, int visit_count);
    
    // ========== Root Statistics Logging (Level 4 - ROOT_STATS and above) ==========
    
    /**
     * @brief Log root node statistics
     * 
     * @param visit_count Total visits to root node
     * @param child_nodes Number of child nodes at root
     */
    void log_root_stats(int visit_count, size_t child_nodes);
    
    /**
     * @brief Log individual child node statistics
     * 
     * @param move Move that lead to the child node
     * @param acc_value Accumulated value for child
     * @param visit_count Visit count for child
     * @param prior_proba Prior probability from neural network
     */
    void log_child_node_stats(const std::array<int, 4>& move,
                              float acc_value, int visit_count, 
                              float prior_proba);
    
    // ========== Final Decision Logging (Level 1 - STEPS_ONLY and above) ==========
    
    /**
     * @brief Log when search time limit is reached
     * 
     * @param iteration_counter Number of iterations completed
     */
    void log_timer_ran_out(int iteration_counter);
    
    /**
     * @brief Log the final move selection
     * 
     * @param iteration_counter Total iterations performed
     * @param move Selected move
     * @param avg_value Average value of selected move
     * @param visits Visit count of selected move
     */
    void log_best_child_chosen(int iteration_counter,
                               const std::array<int, 4>& move,
                               float avg_value, int visits);
    
    // ========== Dirichlet Noise Logging (Level 5 - EVERYTHING) ==========
    
    /**
     * @brief Log application of Dirichlet noise to root
     * 
     * @param alpha Alpha parameter for Dirichlet distribution
     * @param exploration_fraction Fraction of noise to add to priors
     */
    void log_dirichlet_noise_applied(float alpha, float exploration_fraction);
};

#endif // LOGGER_H