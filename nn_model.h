#ifndef ALPHAZERO_MODEL_H
#define ALPHAZERO_MODEL_H

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "cell_state.h"


/**
 * @brief Circular buffer dataset for storing game training data
 * 
 * Stores board states, policy targets, value targets, and legal move masks
 * using a circular buffer strategy for efficient memory usage during training.
 */
struct GameDataset : torch::data::datasets::Dataset<GameDataset> {
    size_t max_size;
    size_t next_index = 0;
    size_t current_size = 0;   
    std::vector<torch::Tensor> boards, pi_targets, z_targets, legal_mask;
    std::unordered_map<std::string, size_t> board_hash_map;

    /**After aggregation, apply temperature and softmax as usual.
     * @brief Create a GameDataset with specified maximum size
     * 
     * @param max_size_ Maximum number of positions to store
     */
    GameDataset(size_t max_size_);

    /**
     * @brief Adds a new training position to the dataset, Check for duplicates
     * 
     * @param board Board state representation
     * @param pi Policy target distribution
     * @param z Value target
     * @param mask Legal move mask
     */
    void add_position(torch::Tensor board, torch::Tensor pi, torch::Tensor z, torch::Tensor mask);

    /**
     * @brief Gets a random training example from the dataset
     * 
     * @return Training example containing board state and concatenated targets
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
     * @brief Save the actual model to disk
     *
     * @param path Path to save the model
     */
    void save(const std::string& path) const;
    
    /**
     * @brief Hash the board into a unique string
     *
     * @param board The tensor of the board to hash
     */
    std::string hash_board(const torch::Tensor& board);
};



/**
 * @brief Neural network architecture for AlphaZero with legal move masking
 * 
 * Implements a residual convolutional neural network with separate policy
 * and value heads for game state evaluation and move prediction.
 */
struct AlphaZeroNetWithMaskImpl : torch::nn::Module {
    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::ModuleList res_blocks{nullptr};
    torch::nn::Conv2d policy_head_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr};
    torch::nn::Conv2d value_head_conv{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};

    int H, W;

    /**
     * @brief Constructs the AlphaZero neural network
     * 
     * @param C Number of input channels
     * @param H_ Board height
     * @param W_ Board width
     * @param num_moves Total number of possible moves
     * @param channels Number of convolutional filters
     * @param n_blocks Number of residual blocks
     */
    AlphaZeroNetWithMaskImpl(int C = 11, int H_ = 3, int W_ = 3, 
                             int num_moves = 9, int channels = 64, int n_blocks = 6);

    /**
     * @brief Forward pass through the network
     * 
     * @param x Input board state tensor
     * @param legal_mask Optional mask for legal moves
     * 
     * @return Pair of policy log-probabilities and value prediction
     */
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, 
                                                     torch::Tensor legal_mask = torch::Tensor());
    
    /**
     * @brief Predict with the network
     * 
     * @param x Input board state tensor
     * @param legal_mask Optional mask for legal moves
     * 
     * @return Pair of policy log-probabilities and value prediction
     */                                                 
    std::pair<torch::Tensor, torch::Tensor> predict(torch::Tensor x, 
                                                     torch::Tensor legal_mask = torch::Tensor());

    /**
     * @brief Saves the model to disk
     * 
     * @param path File path where model will be saved
     */
    void save_model(const std::string& path);

    /**
     * @brief Loads a model from disk
     * 
     * @param path File path from which to load the model
     * 
     * @return Loaded model holder
     */
    static torch::nn::ModuleHolder<AlphaZeroNetWithMaskImpl> load_model(const std::string& path);
};

TORCH_MODULE(AlphaZeroNetWithMask);


/**
 * @brief Computes the AlphaZero combined loss function
 * 
 * Combines cross-entropy loss for policy and mean squared error for value.
 * 
 * @param policy_pred Predicted policy log-probabilities
 * @param value_pred Predicted value
 * @param pi_target Target policy distribution
 * @param z_target Target value
 * 
 * @return Combined loss tensor
 */
torch::Tensor alphazero_loss(torch::Tensor policy_pred, torch::Tensor value_pred,
                             torch::Tensor pi_target, torch::Tensor z_target);


// =============================================================
// ===================== Training Function ======================
// =============================================================

/**
 * @brief Trains the AlphaZero model on the provided dataset
 * 
 * @param model Neural network model to train
 * @param dataset Training dataset containing game positions
 * @param batch_size Number of samples per batch
 * @param epochs Number of training epochs
 * @param lr Learning rate for Adam optimizer
 * @param device Device to train on (CPU or CUDA)
 */
void train(AlphaZeroNetWithMask& model, GameDataset& dataset, int batch_size = 32, 
           int epochs = 5, double lr = 1e-3, torch::Device device = torch::kCUDA);

#endif // ALPHAZERO_MODEL_H
