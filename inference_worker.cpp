#include "inference_queue.h"
#include "alphaz_model.h"
#include <torch/torch.h>
#include <atomic>
#include <iostream>

// Forward declaration
void process_batch(
    std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>>& requests,
    std::shared_ptr<AlphaZModel> model,
    torch::Device device,
    int& C, int& H, int& W, int& action_space,
    bool& dimensions_detected,
    ModelID model_id);

/**
 * Dual-model inference worker
 * Handles inference requests for two models simultaneously
 */
void dual_inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> model1,
    std::shared_ptr<AlphaZModel> model2,
    int batch_size,
    std::atomic<bool>& stop_flag) {
    
    // Validate network pointers
    if (!model1 || !model2) {
        std::cerr << "[Dual Inference Worker] FATAL: model1 or model2 is nullptr!" << std::endl;
        return;
    }
    
    // Check device availability
    bool use_cuda = torch::cuda::is_available();
    torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
    
    std::cout << "[Dual Inference Worker] Device: " << (use_cuda ? "CUDA" : "CPU") << std::endl;
    std::cout << "[Dual Inference Worker] Model 1: " << model1.get() << std::endl;
    std::cout << "[Dual Inference Worker] Model 2: " << model2.get() << std::endl;
    
    const int timeout_ms = 2;
    
    // Set both models to eval mode
    try {
        model1->eval();
        model2->eval();
    } catch (const std::exception& e) {
        std::cerr << "[Dual Inference Worker] FATAL: Failed to set eval mode: "
                  << e.what() << std::endl;
        return;
    }
    
    int C = -1, H = -1, W = -1, action_space = -1;
    bool dimensions_detected = false;
    
    int total_batches_model1 = 0;
    int total_batches_model2 = 0;
    int total_requests_model1 = 0;
    int total_requests_model2 = 0;
    
    while (true) {
        if (stop_flag.load(std::memory_order_acquire)) {
            if (queue.pending_count() == 0) {
                std::cout << "[Dual Inference Worker] Stop flag set and queue empty - exiting"
                          << std::endl;
                break;
            }
        }
        
        // Try to get batches for both models
        std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>> requests_model1;
        std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>> requests_model2;
        
        try {
            requests_model1 = queue.get_batch(batch_size, timeout_ms, ModelID::MODEL_1);
            requests_model2 = queue.get_batch(batch_size, timeout_ms, ModelID::MODEL_2);
        } catch (const std::exception& e) {
            std::cerr << "[Dual Inference Worker] ERROR getting batch: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process Model 1 batch
        if (!requests_model1.empty()) {
            try {
                process_batch(requests_model1, model1, device, C, H, W, action_space, 
                             dimensions_detected, ModelID::MODEL_1);
                total_batches_model1++;
                total_requests_model1 += requests_model1.size();
            } catch (const std::exception& e) {
                std::cerr << "[Dual Inference Worker] ERROR processing Model 1 batch: "
                          << e.what() << std::endl;
                for (auto& request : requests_model1) {
                    try {
                        request->result_promise.set_exception(std::current_exception());
                    } catch (...) {}
                }
            }
        }
        
        // Process Model 2 batch
        if (!requests_model2.empty()) {
            try {
                process_batch(requests_model2, model2, device, C, H, W, action_space, 
                             dimensions_detected, ModelID::MODEL_2);
                total_batches_model2++;
                total_requests_model2 += requests_model2.size();
            } catch (const std::exception& e) {
                std::cerr << "[Dual Inference Worker] ERROR processing Model 2 batch: "
                          << e.what() << std::endl;
                for (auto& request : requests_model2) {
                    try {
                        request->result_promise.set_exception(std::current_exception());
                    } catch (...) {}
                }
            }
        }
        
        // If both batches are empty, we timed out - continue
        if (requests_model1.empty() && requests_model2.empty()) {
            continue;
        }
    }
    
    std::cout << "[Dual Inference Worker] Exiting." << std::endl;
    std::cout << "  Model 1: " << total_batches_model1 << " batches, "
              << total_requests_model1 << " requests" << std::endl;
    std::cout << "  Model 2: " << total_batches_model2 << " batches, "
              << total_requests_model2 << " requests" << std::endl;
}

// Helper function to process a batch for a specific model
void process_batch(
    std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>>& requests,
    std::shared_ptr<AlphaZModel> model,
    torch::Device device,
    int& C, int& H, int& W, int& action_space,
    bool& dimensions_detected,
    ModelID model_id) {
    
    const int actual_batch_size = static_cast<int>(requests.size());
    
    // Detect dimensions from first request if not already done
    if (!dimensions_detected) {
        auto& first_req = requests[0];
        
        if (!first_req->board.defined() || !first_req->mask.defined()) {
            throw std::runtime_error("Undefined board or mask tensor");
        }
        
        auto board_sizes = first_req->board.sizes();
        auto mask_sizes = first_req->mask.sizes();
        
        if (board_sizes.size() != 3) {
            throw std::runtime_error("Invalid board tensor dimensions");
        }
        
        if (mask_sizes.size() != 1) {
            throw std::runtime_error("Invalid mask tensor dimensions");
        }
        
        C = board_sizes[0];
        H = board_sizes[1];
        W = board_sizes[2];
        action_space = mask_sizes[0];
        
        dimensions_detected = true;
        
        std::cout << "[Dual Inference Worker] Detected dimensions: C=" << C
                  << ", H=" << H << ", W=" << W
                  << ", A=" << action_space << std::endl;
    }
    
    // Create batch tensors
    torch::Tensor batch_boards =
        torch::zeros({actual_batch_size, C, H, W}, torch::kFloat32);
    torch::Tensor batch_masks =
        torch::zeros({actual_batch_size, action_space}, torch::kFloat32);
    
    // Fill batch
    for (int i = 0; i < actual_batch_size; ++i) {
        if (requests[i]->board.sizes()[0] != C ||
            requests[i]->board.sizes()[1] != H ||
            requests[i]->board.sizes()[2] != W) {
            throw std::runtime_error("Mismatched board dimensions");
        }
        
        batch_boards[i].copy_(requests[i]->board);
        batch_masks[i].copy_(requests[i]->mask);
    }
    
    // Move to device
    torch::Tensor input_batch = batch_boards.to(device);
    torch::Tensor mask_batch = batch_masks.to(device);
    
    // Run inference
    torch::Tensor policy_batch, value_batch;
    {
        torch::NoGradGuard no_grad;
        auto result = model->predict(input_batch, mask_batch);
        policy_batch = result.first;
        value_batch = result.second;
    }
    
    // Move results back to CPU
    policy_batch = policy_batch.to(torch::kCPU);
    value_batch = value_batch.to(torch::kCPU);
    
    // Set results for each request
    for (int i = 0; i < actual_batch_size; ++i) {
        requests[i]->result_promise.set_value({
            policy_batch[i].clone().detach(),
            value_batch[i].clone().detach()
        });
    }
}

/**
 * Single-model inference worker (for backward compatibility)
 */
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
    int batch_size,
    std::atomic<bool>& stop_flag) {
    
    // For single model, use MODEL_1 ID
    if (!network) {
        std::cerr << "[Inference Worker] FATAL: network is nullptr!" << std::endl;
        return;
    }
    
    bool use_cuda = torch::cuda::is_available();
    torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
    
    std::cout << "[Inference Worker] Device: " << (use_cuda ? "CUDA" : "CPU") << std::endl;
    
    const int timeout_ms = 2;
    
    try {
        network->eval();
    } catch (const std::exception& e) {
        std::cerr << "[Inference Worker] FATAL: Failed to set eval mode: "
                  << e.what() << std::endl;
        return;
    }
    
    int C = -1, H = -1, W = -1, action_space = -1;
    bool dimensions_detected = false;
    
    int total_batches_processed = 0;
    int total_requests_processed = 0;
    
    while (true) {
        if (stop_flag.load(std::memory_order_acquire)) {
            if (queue.pending_count() == 0) {
                std::cout << "[Inference Worker] Stop flag set and queue empty - exiting"
                          << std::endl;
                break;
            }
        }
        
        std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>> requests;
        
        try {
            // Only get MODEL_1 requests for single-model worker
            requests = queue.get_batch(batch_size, timeout_ms, ModelID::MODEL_1);
        } catch (const std::exception& e) {
            std::cerr << "[Inference Worker] ERROR getting batch: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (requests.empty()) {
            continue;
        }
        
        try {
            process_batch(requests, network, device, C, H, W, action_space, 
                         dimensions_detected, ModelID::MODEL_1);
            total_batches_processed++;
            total_requests_processed += requests.size();
        } catch (const std::exception& e) {
            std::cerr << "[Inference Worker] ERROR processing batch: "
                      << e.what() << std::endl;
            
            for (auto& request : requests) {
                try {
                    request->result_promise.set_exception(std::current_exception());
                } catch (...) {}
            }
        }
    }
    
    std::cout << "[Inference Worker] Exiting. Processed "
              << total_batches_processed << " batches, "
              << total_requests_processed << " requests" << std::endl;
}