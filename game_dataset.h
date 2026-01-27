#ifndef GAME_DATASET_H
#define GAME_DATASET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <mutex>
#include "board.h"
#include "metrics_logger.h"

struct GameDatasetAnalysisResults {

    size_t total_positions;
    size_t unique_boards;
    double uniqueness_ratio;
    

    size_t early_game_positions;   // <= 3 pieces
    size_t mid_game_positions;     // 4-9 pieces
    size_t late_game_positions;    // >= 10 pieces
    
    // Game outcomes
    size_t x_wins;
    size_t o_wins;
    size_t draws;
    

    double avg_policy_entropy;
    double avg_legal_moves;
    double max_policy_confidence;
    double avg_policy_confidence;

    double avg_value;
    double value_std;
    double avg_abs_value;
    
    size_t num_games;
    double avg_game_length;
    double min_game_length;
    double max_game_length;
    
    double position_diversity;
    double policy_sharpness;
    
    GameDatasetAnalysisResults() : 
        total_positions(0), unique_boards(0), uniqueness_ratio(0.0),
        early_game_positions(0), mid_game_positions(0), late_game_positions(0),
        x_wins(0), o_wins(0), draws(0),
        avg_policy_entropy(0.0), avg_legal_moves(0.0),
        max_policy_confidence(0.0), avg_policy_confidence(0.0),
        avg_value(0.0), value_std(0.0), avg_abs_value(0.0),
        num_games(0), avg_game_length(0.0), 
        min_game_length(0.0), max_game_length(0.0),
        position_diversity(0.0), policy_sharpness(0.0) {}
};


/**
 * @class GameDataset
 * @brief A dataset for storing and managing game positions for NN training.
 *
 * This class implements a circular buffer dataset that stores game states,
 * policy targets (pi), value targets (z), and legal move masks. Extension from pytorch Dataset
 * class and provides thread-safe operations for concurrent access.
 */
class GameDataset : public torch::data::Dataset<GameDataset> {
public:
    /**
     * @brief Constructs a new GameDataset with specified maximum size.
     * @param max_size_ Maximum number of positions the dataset can hold (default: 100000)
     */
    GameDataset(size_t max_size_ = 10000);

    /**
     * @brief Adds a new position to the dataset.
     * @param board Tensor representing the board state
     * @param pi Tensor representing the policy target (move probabilities)
     * @param z Tensor representing the value target (game outcome)
     * @param mask Tensor representing legal move mask
     */
    void add_position(torch::Tensor board, torch::Tensor pi,
                      torch::Tensor z, torch::Tensor mask);

    /**
     * @brief Retrieves a data sample at the specified index.
     * @param index Index of the sample to retrieve
     * @return torch::data::Example containing the board state and targets
     */
    torch::data::Example<> get(size_t index) override;

    /**
     * @brief Returns the size of the dataset.
     * @return Optional size_t containing the current dataset size
     */
    torch::optional<size_t> size() const override;

    /**
     * @brief Updates the value targets (z) to the true game outcome after the game ends
     * @param new_z_values Vector of new value targets to apply
     * @param winner The final game outcome (Cell_state indicating winner)
     */
    void update_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner);

    /**
     * @brief Saves the dataset to disk.
     * @param path File path where the dataset should be saved
     */
    void save(const std::string& path) const;
    
    /**
     * @brief Saves the dataset to disk.
     * @param path File path where the dataset should be saved
     */
    void load(const std::string& path);

    /**
     * @brief Merges another dataset into this one (Thread-safe)
     * @param other The dataset to merge.
     */
    void merge(const GameDataset& other);

    /**
     * @brief Returns the actual number of positions currently stored.
     */
    size_t actual_size() const { return current_size; }

    /**
     * @brief Clears all data from the dataset.
     */
    void clear();

    /**
     * @brief Launch an full analysis of the dataset
     */
    GameDatasetAnalysisResults analyze() const;

    /**
     * @brief Output a print of the analysis
     */
    void print_analysis() const;
    
    /**
     * @brief Log to file the analysis
     */
    void log_metrics(MetricsLogger& logger, int iteration, const std::string& prefix = "dataset") const;


private:
    std::vector<torch::Tensor> boards;       
    std::vector<torch::Tensor> pi_targets;   
    std::vector<torch::Tensor> z_targets;    
    std::vector<torch::Tensor> legal_mask;   
    size_t max_size;                        
    size_t current_size = 0;
    size_t next_index = 0;             
    mutable std::mutex mutex_;
    
    /**
     * @brief Helper to count pieces of a given board
     */
    int count_pieces(const torch::Tensor& board) const;

    /**
     * @brief Helper to recognize if a given board is empty
     */
    bool is_empty_board(const torch::Tensor& board) const;

    /**
     * @brief Helper to calculate entropy
     */
    double compute_entropy(const torch::Tensor& policy) const;

    /**
     * @brief Hashing function for boards uniqueness
     */
    std::string board_to_hash(const torch::Tensor& board) const;
};

#endif // GAME_DATASET_H