#ifndef GAME_DATASET_H
#define GAME_DATASET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <mutex>
#include "board.h"

class GameDataset : public torch::data::Dataset<GameDataset> {
public:
    GameDataset(size_t max_size_ = 100000);

    void add_position(torch::Tensor board, torch::Tensor pi, 
                     torch::Tensor z, torch::Tensor mask);

    torch::data::Example<> get(size_t index) override;
    
    torch::optional<size_t> size() const override;

    void update_last_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner);

    void save(const std::string& path) const;

    // NEW: Thread-safe merge for parallel self-play
    void merge(const GameDataset& other);

    // NEW: Get actual number of stored positions (not max_size)
    size_t actual_size() const { return current_size; }

    // NEW: Clear the dataset
    void clear();

private:
    std::string hash_board(const torch::Tensor& board) const;

    std::vector<torch::Tensor> boards;
    std::vector<torch::Tensor> pi_targets;
    std::vector<torch::Tensor> z_targets;
    std::vector<torch::Tensor> legal_mask;
    
    size_t max_size;
    size_t current_size = 0;
    size_t next_index = 0;

    // NEW: Mutex for thread-safe operations
    mutable std::mutex mutex_;
};

#endif // GAME_DATASET_H