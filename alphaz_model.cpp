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
    

    policy_head_conv = register_module("policy_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 2, 1)));
    policy_head_bn = register_module("policy_head_bn", 
        torch::nn::BatchNorm2d(2));
    policy_fc = register_module("policy_fc", 
        torch::nn::Linear(2 * H * W, num_moves));
    

    value_head_conv = register_module("value_head_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 1, 1)));
    value_head_bn = register_module("value_head_bn", 
        torch::nn::BatchNorm2d(1));
    value_fc1 = register_module("value_fc1", 
        torch::nn::Linear(H * W, 64));
    value_fc2 = register_module("value_fc2", 
        torch::nn::Linear(64, 1));
}

std::pair<torch::Tensor, torch::Tensor> AlphaZModel::forward(
    torch::Tensor x, torch::Tensor legal_mask) {
    
    x = torch::relu(conv_in_bn->forward(conv_in->forward(x)));
    
    // Residual blocks
    for (size_t i = 0; i < res_blocks->size(); i += 4) {
        auto residual = x.clone();
        
        x = res_blocks[i]->as<torch::nn::Conv2d>()->forward(x);
        x = res_blocks[i+1]->as<torch::nn::BatchNorm2d>()->forward(x);
        x = torch::relu(x);
        
        x = res_blocks[i+2]->as<torch::nn::Conv2d>()->forward(x);
        x = res_blocks[i+3]->as<torch::nn::BatchNorm2d>()->forward(x);
        
        x = torch::relu(x + residual);  // Add residual
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
    bool detailed_logging,
    int current_iteration,
    int global_step,
    MetricsLogger& logger
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
    double total_policy_entropy = 0.0;
    double total_value_accuracy = 0.0;
    double max_gradient_norm = 0.0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
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
        
        auto policy_loss_tensor = -(pi_target * p).sum(1).mean();
        auto value_loss_tensor = torch::mse_loss(v, z_target);
        auto total_loss_tensor = policy_loss_tensor + value_loss_tensor;
        
        // Additional metrics for better monitoring
        auto policy_entropy = -(p * torch::log(p + 1e-8)).sum(1).mean();
        auto value_accuracy = (torch::abs(v - z_target) < 0.4).to(torch::kFloat).mean();
        auto sign_accuracy = ((v * z_target) > 0).to(torch::kFloat).mean();
        
        total_policy_entropy += policy_entropy.item<double>();
        total_value_accuracy += value_accuracy.item<double>();
        
        // Backward pass
        total_loss_tensor.backward();
        
        // Gradient clipping for stability
        double grad_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        max_gradient_norm = std::max(max_gradient_norm, grad_norm);
        
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
        
        if (step % 1 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();
            
            double avg_policy_loss = total_policy_loss / log_interval;
            double avg_value_loss = total_value_loss / log_interval;
            double avg_total_loss = total_loss / log_interval;
            double avg_policy_entropy = total_policy_entropy / log_interval;
            double avg_value_accuracy = total_value_accuracy / log_interval;
            
            // Calculate training throughput
            double samples_per_sec = (step * batch_size) / static_cast<double>(std::max(elapsed, 1L));
            
            // Get current learning rate
            double lr_value = 0.0;
            for (const auto& group : optimizer.param_groups()) {
                lr_value = group.options().get_lr();
                break;
            }
            
            // Console logging
            if (detailed_logging) {
                std::cout << "Step " << step << "/" << training_steps 
                          << " | Total Loss: " << std::fixed << std::setprecision(4) << avg_total_loss
                          << " | Policy Loss: " << avg_policy_loss
                          << " | Value Loss: " << avg_value_loss
                          << " | Entropy: " << avg_policy_entropy
                          << " | V-Acc: " << std::setprecision(3) << value_accuracy.item<double>()
                          << " | WinnerPred- Accuracy: " << std::setprecision(3) << sign_accuracy.item<double>()
                          << " | LR: " << std::scientific << std::setprecision(2) << lr_value
                          << " | GradNorm: " << std::fixed << std::setprecision(3) << max_gradient_norm
                          << std::endl;
            } else {
                std::cout << "Step " << step << "/" << training_steps 
                          << " | Loss: " << std::fixed << std::setprecision(4) << avg_total_loss
                          << " | LR: " << std::scientific << std::setprecision(2) << lr_value
                          << std::endl;
            }
            
            // Log to CSV for W&B upload
            int current_global_step = global_step + step;
            
            logger.add_scalar("global_step", current_global_step);
            logger.add_scalar("iteration", current_iteration);
            logger.add_scalar("training_step", step);
            logger.add_scalar("training/total_loss", total_loss_tensor.item<double>());
            logger.add_scalar("training/policy_loss", policy_loss_tensor.item<double>());
            logger.add_scalar("training/value_loss", value_loss_tensor.item<double>());
            logger.add_scalar("training/policy_entropy", policy_entropy.item<double>());
            logger.add_scalar("training/value_accuracy", value_accuracy.item<double>());
            logger.add_scalar("training/learning_rate", lr_value);
            logger.add_scalar("training/gradient_norm", max_gradient_norm);
            logger.add_scalar("training/samples_per_sec", samples_per_sec);
            logger.add_scalar("training/sign_accuracy", sign_accuracy.item<double>());
            logger.flush_metrics();
            
            // Reset accumulators
            total_policy_loss = 0.0;
            total_value_loss = 0.0;
            total_loss = 0.0;
            total_policy_entropy = 0.0;
            total_value_accuracy = 0.0;
            max_gradient_norm = 0.0;
        }
    }
    
    std::cout << "\n✅ Training completed!\n" << std::endl;
}