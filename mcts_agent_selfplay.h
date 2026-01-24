#ifndef MCTS_AGENT_SELFPLAY_H
#define MCTS_AGENT_SELFPLAY_H

#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "board.h"
#include "inference_queue.h"
#include "logger.h"


/**
 * @brief Implements a Monte Carlo Tree Search (MCTS) agent for decision-making in games
 *
 * Simulates gameplay guided by a neural network to determine the best move within
 * a fixed number of iterations. Balances exploration and exploitation using the
 * PUCT formula with neural network priors. Supports optional logging.
 *
 * @note Assumes a `Board` class with `get_valid_moves()`, `make_move()`, and
 *       `check_winner()` methods, and a `Cell_state` enum with `Empty`, `X`, and `O`.
 */

class Mcts_agent_selfplay {
public:
    /**
     * @brief Constructs a new MCTS agent
     *
     * @param exploration_factor Exploration factor for MCTS (PUCT constant)
     * @param number_iteration Maximum iteration number for MCTS simulations
     * @param log_level Log Level for debugging output
     * @param temperature Temperature parameter for action selection (default: 1.0)
     * @param dirichlet_alpha Alpha parameter for Dirichlet noise (default: 0.3)
     * @param dirichlet_epsilon Epsilon for mixing Dirichlet noise with policy (default: 0.25)
     * @param network Neural network model for policy/value evaluation (default: nullptr)
     * @param max_depth Maximum search depth for MCTS (default: -1, unlimited)
     * @param tree_reuse Whether to reuse the search tree between moves (default: false)
     * @param model_id Indicator for the model to use for inference
    */
    Mcts_agent_selfplay(double exploration_factor,
              int number_iteration,
              LogLevel log_level = LogLevel::NONE,
              float temperature = 1.0f,
              float dirichlet_alpha = 0.3f,
              float dirichlet_epsilon = 0.25f,
              InferenceQueue* inference_queue = nullptr,
              int max_depth = -1,
              bool tree_reuse = false,
              ModelID model_id = ModelID::MODEL_1);

    /**
     * @brief Selects the best move using Monte Carlo Tree Search (MCTS)
     *
     * Performs neural network-guided MCTS simulations from the current state,
     * updating node statistics to identify the optimal move. The process runs
     * for the specified number of iterations, then selects the best child based
     * on visit counts.
     *
     * Verbose mode based on levels to log MCTS statistics.
     *
     * @param board Current game state
     * @param player The player making the move
     *
     * @return Pair containing the best move as std::array<int, 4> and policy tensor
     *
     * @throws runtime_error If insufficient simulations prevent a reliable decision
     */
    std::pair<std::array<int, 4>, torch::Tensor> choose_move(const Board& board, Cell_state player);

    /**
     * @brief Executes random moves on the board for exploration
     *network
     * Updates the board state by making a specified number of random valid moves,
     * alternating between players as appropriate.
     *
     * @param board Board state to modify
     * @param player The current player making the first move
     * @param random_move_number Number of random moves to execute
     */
    void random_move(Board& board, Cell_state player, int random_move_number);

private:
    InferenceQueue* inference_queue = nullptr;
    double exploration_factor;
    int number_iteration;
    LogLevel log_level;
    std::shared_ptr<Logger> logger;
    std::random_device random_device;
    std::mt19937 random_generator; 
    float temperature;          // Temperature for move selection
    float dirichlet_alpha;      // Alpha parameter for Dirichlet noise
    float dirichlet_epsilon;    // Epsilon for Dirichlet noise mixing
    int max_depth;              // Maximum search depth
    bool tree_reuse;            // Whether to reuse search tree
    ModelID model_id;

    /**
     * @brief Represents a node in the Monte Carlo Tree Search (MCTS) tree
     *
     * Each node corresponds to a unique game state, storing both the state
     * information and statistics accumulated during the search process.
     */
    struct Node {
        /**
         * @brief Value estimate from the neural network evaluation
         */
        float value_from_nn;

        /**
         * @brief Mean action value from MCTS simulations (Q-value)
         */
        float value_from_mcts;

        /**
         * @brief Flag indicating if the node has been expanded with neural network evaluation
         */
        bool expanded;

        /**
         * @brief Accumulated value from all simulations passing through this node
         */
        float acc_value;

        /**
         * @brief Prior policy probability from the neural network
         */
        float prior_proba;

        /**
         * @brief Number of times this node has been visited during search
         */
        int visit_count;

        /**
         * @brief Move that led to this state from parent (array format: {x, y, direction, target})
         *
         * For root node, uses sentinel value {-1, -1, -1, -1}
         */
        std::array<int, 4> move;

        /**
         * @brief Player that will make the move from this state
         */
        Cell_state player;

        /**
         * @brief Child nodes representing states reachable by one move
         */

        std::vector<std::shared_ptr<Node>> child_nodes;

        /**
         * @brief Pointer to parent node (nullptr for root)
         */
        std::shared_ptr<Node> parent_node;

        /**
         * @brief Mutex for thread-safe node updates during parallel MCTS
         */
        std::mutex node_mutex;

        /**
         * @brief Create a new MCTS tree node
         *
         * @param player Player making the move from this state
         * @param move Move array {x, y, direction, target} leading to this state
         * @param prior_proba Prior probability of the move leading to this sate from neural network policy
         * @param value_from_nn Value estimate of this state from neural network
         * @param parent_node Parent node pointer (nullptr for root)
         */
        Node(Cell_state player, std::array<int, 4> move, float prior_proba, 
             float value_from_nn, std::shared_ptr<Node> parent_node = nullptr);
    };

    std::shared_ptr<Node> root;

    /**
     * @brief Initializes node and evaluates it with the neural network
     *
     * Expands the given node by creating child nodes for all valid moves.
     * Queries the neural network for policy priors and value estimate.
     * Optionally adds Dirichlet noise to root node for exploration.
     *
     * @param node Node to initialize and expand
     * @param board Current game state
     * @param add_dirichlet_noise Whether to add exploration noise to priors
     * @param dirichlet_alpha Concentration parameter for Dirichlet distribution
     * @param exploration_fraction Weight of noise vs network priors (0.0 - 1.0)
     *
     * @return Value estimate from neural network for this position
     */
    float initiate_and_run_nn(const std::shared_ptr<Node>& node,
                               const Board& board,
                               bool add_dirichlet_noise,
                               float dirichlet_alpha,
                               float exploration_fraction);

    /**
     * @brief Generates Dirichlet noise for exploration at root node
     *
     * @param num_moves Number of legal moves to generate noise for
     * @param alpha Concentration parameter controlling noise distribution
     *
     * @return Vector of noise values, one per legal move
     */
    std::vector<float> generate_dirichlet_noise(int num_moves, float alpha);

    /**
     * @brief Performs Monte Carlo Tree Search guided by neural network
     *
     * Executes the main MCTS loop for a specified number of iterations.
     * Each iteration selects, expands, simulates (via NN), and backpropagates.
     * Logs statistics according to verbose mode level
     *
     * @param number_iteration Number of MCTS simulations to run
     * @param mcts_iteration_counter Reference to iteration counter for logging
     * @param board Initial game state to search from
     */
    void perform_mcts_iterations(const int number_iteration,
                                  int& mcts_iteration_counter,
                                  const Board& board);

    /**
    * @brief Sample the next move based on visit counts and temperature.
    *
    * Generates a tensor representing the search statistics for all
    * possible moves. Each entry corresponds to the visit count of each child
    * node, adjusted by temperature and normalized.
    *
    * @param parent_node Node whose children are evaluated for policy extraction
    * @param temperature Controls exploration
    *
    * @return Pair containing the selected child node and a 1D policy tensor of size 
    *         (X * Y * DIR * TAR) for data collection.
    */
    std::pair<std::shared_ptr<Mcts_agent_selfplay::Node>, torch::Tensor> sample_child_and_get_policy(
    const std::shared_ptr<Node>& parent_node) const;

    /**
     * @brief Converts policy tensor into list of moves with probabilities
     *
     * Extracts non-zero entries from the policy tensor and converts them into
     * a list of move-probability pairs for easier processing.
     *
     * @param log_probs_tensor 1D tensor of log probabilities for all moves from NN
     *
     * @return Vector of pairs: (move array {x, y, direction, target}, normalized probability)
     */
    std::vector<std::pair<std::array<int, 4>, float>> get_moves_with_probs(
        const torch::Tensor& log_probs_tensor) const;

    /**
     * @brief Select a leaf by moving the tree using PUCT Score
     *
     * Evaluates all children of the parent node and selects the one with the
     * highest PUCT score, which balances exploitation (Q-value) and exploration
     * (prior probability and visit counts). Updates the board state accordingly.
     *
     * @param parent_node Node where to start selection
     * @param board Current board state (will be modified with selected move)
     *
     * @return Pair of (selected child node, corresponding board state)
     */
    std::pair<std::shared_ptr<Mcts_agent_selfplay::Node>, Board> select_child_for_playout(
        const std::shared_ptr<Node>& parent_node, Board board);

    /**
     * @brief Computes the PUCT score
     *
     * Calculates the PUCT score used in AlphaZero-style MCTS to balance
     * exploitation (mean action value) and exploration (prior probability
     * weighted by visit counts). Unvisited nodes return a high value to
     * encourage exploration.
     *
     * Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
     *
     * @param child_node Child node being evaluated
     * @param parent_node Parent node providing context for visit count
     *
     * @return PUCT score for the child node
     */
    double calculate_puct_score(const std::shared_ptr<Node>& child_node,
                                const std::shared_ptr<Node>& parent_node);

    /**
     * @brief Simulates random playout from a given node
     *
     * Plays out the game from the current state using random valid moves
     * until a terminal state is reached. Used for evaluation when neural
     * network guidance is not available.
     *
     * Logs statistics according to verbose mode level
     *
     * @param node Node at which to start selection
     * @param board Board state to simulate from (copied, original unchanged)
     *
     * @return Game outcome value from the perspective of the node's player
     */
    float simulate_random_playout(const std::shared_ptr<Node>& node, Board board);

    /**
     * @brief Backpropagates simulation results through the MCTS tree
     *
     * Updates visit counts and accumulated values from the given node up to
     * the root. The value is propagated with sign flips at each level to
     * maintain proper perspective for alternating players. Thread-safe via
     * mutex locks.
     *
     * @param node Node at which to start backpropagation
     * @param value Outcome value to backpropagate (-1 to 1 scale)
     */
    void backpropagate(std::shared_ptr<Node>& node, float value);

};

#endif