#ifndef GAME_DATASET_H
#define GAME_DATASET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <mutex>
#include "board.h"

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

private:
    std::vector<torch::Tensor> boards;       
    std::vector<torch::Tensor> pi_targets;   
    std::vector<torch::Tensor> z_targets;    
    std::vector<torch::Tensor> legal_mask;   
    size_t max_size;                        
    size_t current_size = 0;
    size_t next_index = 0;             
    mutable std::mutex mutex_;
};

#endif // GAME_DATASET_H