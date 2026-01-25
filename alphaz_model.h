#ifndef ALPHAZ_MODEL_H
#define ALPHAZ_MODEL_H

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <iomanip>
#include "game_dataset.h"

/**
 * @brief Neural network architecture for AlphaZero with legal move masking
 * 
 * Implements a residual convolutional neural network with separate policy
 * and value heads for game state evaluation and move prediction.
 */
class AlphaZModel : public torch::nn::Module {
public:
    torch::nn::Conv2d conv_in{nullptr};
    torch::nn::BatchNorm2d conv_in_bn{nullptr};
    torch::nn::ModuleList res_blocks{nullptr};
    torch::nn::Conv2d policy_head_conv{nullptr};
    torch::nn::BatchNorm2d policy_head_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr};
    torch::nn::Conv2d value_head_conv{nullptr};
    torch::nn::BatchNorm2d value_head_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};

    int H, W;

    /**
     * @brief Constructs the AlphaZ neural network
     * 
     * @param C Number of input channels
     * @param H_ Board height
     * @param W_ Board width
     * @param num_moves Total number of possible moves
     * @param channels Number of convolutional filters
     * @param n_blocks Number of residual blocks
     */
    AlphaZModel(int C = 3, int H_ = 4, int W_ = 4, 
                int num_moves = 16, int channels = 64, int n_blocks = 3);

    /**
     * @brief Forward pass through the network
     * 
     * @param x Input board state tensor
     * @param legal_mask legal mask for legal moves
     * 
     * @return Pair of policy log-probabilities and value prediction
     */
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, 
                                                     torch::Tensor legal_mask = torch::Tensor());
    
    /**
     * @brief Predict with the network (inference mode, no gradients)
     * 
     * @param x Input board state tensor
     * @param legal_mask Optional mask for legal moves
     * 
     * @return Pair of policy log-probabilities and value prediction
     */                                                 
    std::pair<torch::Tensor, torch::Tensor> predict(torch::Tensor x, 
                                                     torch::Tensor legal_mask = torch::Tensor());

    /**
     * @brief Saves the model parameters to disk
     * 
     * @param path File path where model parameters will be saved
     */
    void save_model(const std::string& path);

    /**
     * @brief Loads a model from disk
     * 
     * @param path File path from which to load the model parameters
     * @param C Number of input channels
     * @param H_ Board height
     * @param W_ Board width
     * @param num_moves Total number of possible moves
     * @param channels Number of convolutional filters
     * @param n_blocks Number of residual blocks
     * 
     * @return Loaded model with parameters restored
     */
    static std::shared_ptr<AlphaZModel> load_model(
        const std::string& path,
        int C = 3, int H_ = 4, int W_ = 4,
        int num_moves = 16, int channels = 64, int n_blocks = 3);
};

/**
 * @brief Calculates the AlphaZero combined loss function
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

/**
 * @brief Trains the AlphaZ model on the provided dataset
 * 
 * Implements training loop with Adam optimizer, cosine annealing learning rate schedule,
 * gradient clipping, and periodic logging of training metrics.
 * 
 * @param model Neural network model to train
 * @param dataset Training dataset containing game positions
 * @param batch_size Number of samples per batch
 * @param training_steps Total number of training steps
 * @param lr Initial learning rate for Adam optimizer
 * @param device Device to train on (CPU or CUDA)
 * @param log_interval Number of steps between metric logging
 * @param detailed_logging Print detailed loss components (policy/value) or not
 */
void train(std::shared_ptr<AlphaZModel> model, GameDataset& dataset, 
           int batch_size = 32, int training_steps = 1000, 
           double lr = 1e-3, torch::Device device = torch::kCUDA,
           int log_interval = 5, bool detailed_logging = false);

#endif // ALPHAZ_MODEL_H