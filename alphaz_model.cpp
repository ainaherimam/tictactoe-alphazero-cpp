#include "alphaz_model.h"
#include <random>
#include <cmath>
#include <iomanip>
#include <iostream>

AlphaZModel::AlphaZModel(int C, int H_, int W_, 
                         int num_moves, int channels, int n_blocks) 
    : H(H_), W(W_) {
    
    
    conv_in = register_module("conv_in", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(C, channels, 3).padding(1)));
    conv_in_bn = register_module("conv_in_bn", 
        torch::nn::BatchNorm2d(channels));
    
    res_blocks = register_module("res_blocks", torch::nn::ModuleList());
    
    for (int i = 0; i < n_blocks; ++i) {
        auto conv1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1));
        auto bn1 = torch::nn::BatchNorm2d(channels);
        auto conv2 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1));
        auto bn2 = torch::nn::BatchNorm2d(channels);
        
        res_blocks->push_back(
            register_module("res_conv1_" + std::to_string(i), conv1));
        res_blocks->push_back(
            register_module("res_bn1_" + std::to_string(i), bn1));
        res_blocks->push_back(
            register_module("res_conv2_" + std::to_string(i), conv2));
        res_blocks->push_back(
            register_module("res_bn2_" + std::to_string(i), bn2));
    }
    
    // Policy head
    policy_head_conv = register_module("policy_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 2, 1)));
    policy_head_bn = register_module("policy_head_bn", 
        torch::nn::BatchNorm2d(2));
    policy_fc = register_module("policy_fc", 
        torch::nn::Linear(2 * H * W, num_moves));
    
    // Value head (smaller for 4x4)
    value_head_conv = register_module("value_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 1, 1)));
    value_head_bn = register_module("value_head_bn", 
        torch::nn::BatchNorm2d(1));
    value_fc1 = register_module("value_fc1", 
        torch::nn::Linear(H * W, 64));  // 64 is fine for 4x4
    value_fc2 = register_module("value_fc2", 
        torch::nn::Linear(64, 1));
}

std::pair<torch::Tensor, torch::Tensor> AlphaZModel::forward(
    torch::Tensor x, torch::Tensor legal_mask) {
    
    // Initial convolution
    x = torch::relu(conv_in_bn->forward(conv_in->forward(x)));
    
    // Residual blocks (proper implementation)
    for (size_t i = 0; i < res_blocks->size(); i += 4) {
        auto residual = x.clone();
        
        x = res_blocks[i]->as<torch::nn::Conv2d>()->forward(x);
        x = res_blocks[i+1]->as<torch::nn::BatchNorm2d>()->forward(x);
        x = torch::relu(x);
        
        x = res_blocks[i+2]->as<torch::nn::Conv2d>()->forward(x);
        x = res_blocks[i+3]->as<torch::nn::BatchNorm2d>()->forward(x);
        
        x = torch::relu(x + residual);  // Add residual, then ReLU
    }
    
    // Policy Head
    auto p = policy_head_conv->forward(x);
    p = torch::relu(policy_head_bn->forward(p));
    p = p.view({x.size(0), -1});
    p = policy_fc->forward(p);
    
    if (legal_mask.defined()) {
        p = p.masked_fill(legal_mask == 0, -1e9);
    }
    p = torch::log_softmax(p, 1);
    
    // Value Head
    auto v = value_head_conv->forward(x);
    v = torch::relu(value_head_bn->forward(v));
    v = v.view({x.size(0), -1});
    v = torch::relu(value_fc1->forward(v));
    v = torch::tanh(value_fc2->forward(v)).squeeze(-1);
    
    return {p, v};
}

std::pair<torch::Tensor, torch::Tensor> AlphaZModel::predict(torch::Tensor x, 
                                                              torch::Tensor legal_mask) {
    torch::NoGradGuard no_grad;
    return this->forward(x, legal_mask);
}

void AlphaZModel::save_model(const std::string& path) {
    try {
        torch::save(std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this()), path);
        std::cout << "✅ Model saved to: " << path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "❌ Error saving model: " << e.msg() << std::endl;
    }
}

std::shared_ptr<AlphaZModel> AlphaZModel::load_model(
    const std::string& path,
    int C, int H_, int W_,
    int num_moves, int channels, int n_blocks) {
    try {
        auto model = std::make_shared<AlphaZModel>(C, H_, W_, num_moves, channels, n_blocks);
        torch::load(model, path);
        return model;
    } catch (const c10::Error& e) {
        std::cerr << "❌ Error loading model: " << e.msg() << std::endl;
        throw;
    }
}

torch::Tensor alphazero_loss(torch::Tensor policy_pred, torch::Tensor value_pred,
                             torch::Tensor pi_target, torch::Tensor z_target) {

    auto policy_loss = -(pi_target * policy_pred).sum(1).mean();
    auto value_loss = torch::mse_loss(value_pred, z_target);

    //MAYBE L2 REGULARIZATION HERE!!!
    return policy_loss + value_loss;
}

void train(
    std::shared_ptr<AlphaZModel> model,
    GameDataset& dataset,
    int batch_size,
    int training_steps,
    double lr,
    torch::Device device,
    int log_interval,
    bool detailed_logging
) {
    model->to(device);
    model->train();
    
    // Optimizer with weight decay (L2 regularization)
    torch::optim::Adam optimizer(
        model->parameters(), 
        torch::optim::AdamOptions(lr).weight_decay(1e-4)
    );
    
    double initial_lr = lr;
    double min_lr = 1e-4;
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, dataset.size().value() - 1);
    
    double total_policy_loss = 0.0;
    double total_value_loss = 0.0;
    double total_loss = 0.0;
    
    for (int step = 1; step <= training_steps; ++step) {
        std::vector<torch::Tensor> states, targets;
        
        // Sample batch
        for (int i = 0; i < batch_size; ++i) {
            auto idx = dist(rng);
            auto sample = dataset.get(idx);
            states.push_back(sample.data);
            targets.push_back(sample.target);
        }
        
        auto b = torch::stack(states).to(device);
        auto t = torch::stack(targets).to(device);
        auto pi_target = t.slice(1, 0, 16);
        auto z_target  = t.slice(1, 16, 17).squeeze(1);
        auto mask      = t.slice(1, 17, t.size(1));
        
        // Forward pass
        optimizer.zero_grad();
        auto [p, v] = model->forward(b, mask);
        
        // Compute losses
        auto policy_loss_tensor = -(pi_target * p).sum(1).mean();
        auto value_loss_tensor = torch::mse_loss(v, z_target);
        auto total_loss_tensor = policy_loss_tensor + value_loss_tensor;
        
        // Backward pass
        total_loss_tensor.backward();
        
        // Optional: Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        
        optimizer.step();
        // Update learning rate using cosine annealing
        double current_lr = min_lr + (initial_lr - min_lr) * 
                           (1 + std::cos(M_PI * step / training_steps)) / 2.0;
        
        for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(current_lr);
        }
        
        total_policy_loss += policy_loss_tensor.item<double>();
        total_value_loss += value_loss_tensor.item<double>();
        total_loss += total_loss_tensor.item<double>();
        
        // Log progress
        if (step % log_interval == 0) {
            double avg_policy_loss = total_policy_loss / log_interval;
            double avg_value_loss = total_value_loss / log_interval;
            double avg_total_loss = total_loss / log_interval;
            
            // Get current learning rate
            double current_lr = 0.0;
            for (const auto& group : optimizer.param_groups()) {
                current_lr = group.options().get_lr();
                break;
            }
            
            if (detailed_logging) {
                std::cout << "Step " << step << "/" << training_steps 
                          << " | Total Loss: " << std::fixed << std::setprecision(4) << avg_total_loss
                          << " | Policy Loss: " << avg_policy_loss
                          << " | Value Loss: " << avg_value_loss
                          << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                          << std::endl;
            } else {
                std::cout << "Step " << step << "/" << training_steps 
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_total_loss
                          << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                          << std::endl;
            }
            
            // Reset accumulators
            total_policy_loss = 0.0;
            total_value_loss = 0.0;
            total_loss = 0.0;
        }
    }
    
    std::cout << "\n✅ Training completed!\n" << std::endl;
}