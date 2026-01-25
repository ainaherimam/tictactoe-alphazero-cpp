#include "game_dataset.h"
#include <random>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <algorithm>

GameDataset::GameDataset(size_t max_size_) : max_size(max_size_) {
    boards.resize(max_size);
    pi_targets.resize(max_size);
    z_targets.resize(max_size);
    legal_mask.resize(max_size);
}

void GameDataset::add_position(torch::Tensor board, torch::Tensor pi, 
                               torch::Tensor z, torch::Tensor mask) {
    std::lock_guard<std::mutex> lock(mutex_);
    

    boards[next_index] = board;
    pi_targets[next_index] = pi;
    z_targets[next_index] = z;
    legal_mask[next_index] = mask;
    
    // Circular buffer advancement
    next_index = (next_index + 1) % max_size;
    if (current_size < max_size) {
        current_size++;
    }
}

torch::data::Example<> GameDataset::get(size_t) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    static std::mt19937 rng(std::random_device{}());
    size_t idx = dist(rng);
    
    return {boards[idx], torch::cat({pi_targets[idx], z_targets[idx].unsqueeze(0), legal_mask[idx]})};
}

torch::optional<size_t> GameDataset::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_size;
}

void GameDataset::update_z(const std::vector<torch::Tensor>& new_z_values, Cell_state winner) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = new_z_values.size();
    
    // Compute z values for X and O based on winner
    float z_val_x = (winner == Cell_state::X) ? 1.0f : (winner == Cell_state::O ? -1.0f : 0.0f);
    float z_val_o = (winner == Cell_state::O) ? 1.0f : (winner == Cell_state::X ? -1.0f : 0.0f);
    
    for (size_t i = 0; i < count; ++i) {
        size_t idx = (next_index + max_size - count + i) % max_size;
        float old_val = z_targets[idx].item<float>();
        float updated_val = (old_val == 0.0f) ? z_val_x : z_val_o;
        z_targets[idx] = torch::tensor(updated_val, torch::dtype(torch::kFloat32));
    }
}

void GameDataset::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    torch::save(boards, path + "_boards.pt");
    torch::save(pi_targets, path + "_pi.pt");
    torch::save(z_targets, path + "_z.pt");
    torch::save(legal_mask, path + "_mask.pt");
}

//Thread-safe merge for parallel self-play
void GameDataset::merge(const GameDataset& other) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < other.current_size; ++i) {

        boards[next_index] = other.boards[i].clone();
        pi_targets[next_index] = other.pi_targets[i].clone();
        z_targets[next_index] = other.z_targets[i].clone();
        legal_mask[next_index] = other.legal_mask[i].clone();
        
        // Advance circular buffer
        next_index = (next_index + 1) % max_size;
        if (current_size < max_size) {
            current_size++;
        }
    }
}

// Clear the dataset
void GameDataset::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_size = 0;
    next_index = 0;

    // boards.clear();
    // pi_targets.clear();
    // z_targets.clear();
    // legal_mask.clear();
    // boards.resize(max_size);
    // pi_targets.resize(max_size);
    // z_targets.resize(max_size);
    // legal_mask.resize(max_size);
}