#ifndef GAME_DATASET_H
#define GAME_DATASET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "cell_state.h"

struct DatasetAnalysis {
    size_t total_states;
    size_t unique_states;
    double uniqueness_ratio;
    std::unordered_map<std::string, int> state_frequency;
    std::vector<int> repetition_counts;
    double mean_entropy;
    double std_entropy;
    int mismatched_terminal_states;
    
    // Game length tracking (approximate based on z-value transitions)
    std::vector<int> game_lengths;
    double mean_game_length;
    double std_game_length;
    int min_game_length;
    int max_game_length;
};

/**
 * @brief Circular buffer dataset for storing game training data
 * 
 * Stores board states, legal move masks, policy targets and value targets
 * using a circular buffer.
 */
struct GameDataset : torch::data::datasets::Dataset<GameDataset> {
    size_t max_size;
    size_t next_index = 0;
    size_t current_size = 0;   
    std::vector<torch::Tensor> boards, pi_targets, z_targets, legal_mask;
    std::unordered_map<std::string, size_t> board_hash_map;

    /**
     * @brief Create a GameDataset with specified maximum size
     * 
     * @param max_size_ Maximum number of positions to store
     */
    GameDataset(size_t max_size_);

    /**
     * @brief Adds a new training position to the dataset.
     * 
     * @param board Board state representation
     * @param pi Policy target distribution
     * @param z Value target
     * @param mask Legal move mask
     */
    void add_position(torch::Tensor board, torch::Tensor pi, torch::Tensor z, torch::Tensor mask);

    /**
     * @brief Gets a random training move sample from the dataset
     * 
     * @return Training example containing board state and targets
     */
    torch::data::Example<> get(size_t index) override;

    /**
     * @brief Returns the current size of the dataset
     * 
     * @return Optional size value
     */
    torch::optional<size_t> size() const override;

    /**
     * @brief Updates the value targets for the most recent game positions
     * 
     * @param new_z_values Vector of new value targets
     * @param winner Game outcome (X, O, or draw)
     */
    void update_last_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner);

    /**
     * @brief Save the dataset to disk
     *
     * @param path Path to save the dataset
     */
    void save(const std::string& path) const;

    void print_analysis() const;

    DatasetAnalysis analyze_dataset() const;
    
    /**
     * @brief Hash the board into a unique string
     *
     * @param board The tensor of the board to hash
     */
    std::string hash_board(const torch::Tensor& board) const ;
};

#endif // GAME_DATASET_H