#include "inference_queue.h"
#include "alphaz_model.h"
#include <torch/torch.h>
#include <atomic>
#include <iostream>

/**
 * CRITICAL FIX FOR CPU MODE
 * The previous version may have issues when CUDA is not available.
 * This version is more defensive about tensor operations.
 */
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
    int batch_size,
    std::atomic<bool>& stop_flag) {
    
    // std::cout << "[Inference Worker] Thread started, ID = "
    //           << std::this_thread::get_id() << std::endl;
    
    // CRITICAL: Validate network pointer immediately
    if (!network) {
        std::cerr << "[Inference Worker] FATAL: network is nullptr!" << std::endl;
        return;
    }
    
    // std::cout << "[Inference Worker] Network pointer valid: " << network.get() << std::endl;
    
    // CRITICAL: Check device availability
    bool use_cuda = torch::cuda::is_available();
    torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
    
    std::cout << "[Inference Worker] Device: " << (use_cuda ? "CUDA" : "CPU") << std::endl;
    
    const int timeout_ms = 2;
    
    // CRITICAL: Validate network is properly initialized
    try {
        // std::cout << "[Inference Worker] Setting network to eval mode..." << std::endl;
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
    
    // std::cout << "[Inference Worker] Entering main loop (batch_size="
    //           << batch_size << ")" << std::endl;
    
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
            requests = queue.get_batch(batch_size, timeout_ms);
        } catch (const std::exception& e) {
            std::cerr << "[Inference Worker] ERROR getting batch: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (requests.empty()) {
            continue;
        }
        
        const int actual_batch_size = static_cast<int>(requests.size());
        
        try {
            // CRITICAL: Detect dimensions from first request
            if (!dimensions_detected) {
                auto& first_req = requests[0];
                
                if (!first_req->board.defined()) {
                    std::cerr << "[Inference Worker] ERROR: Undefined board tensor!" << std::endl;
                    throw std::runtime_error("Undefined board tensor");
                }
                
                if (!first_req->mask.defined()) {
                    std::cerr << "[Inference Worker] ERROR: Undefined mask tensor!" << std::endl;
                    throw std::runtime_error("Undefined mask tensor");
                }
                
                auto board_sizes = first_req->board.sizes();
                auto mask_sizes = first_req->mask.sizes();
                
                if (board_sizes.size() != 3) {
                    std::cerr << "[Inference Worker] ERROR: Expected 3D board tensor, got "
                              << board_sizes.size() << "D" << std::endl;
                    throw std::runtime_error("Invalid board tensor dimensions");
                }
                
                if (mask_sizes.size() != 1) {
                    std::cerr << "[Inference Worker] ERROR: Expected 1D mask tensor, got "
                              << mask_sizes.size() << "D" << std::endl;
                    throw std::runtime_error("Invalid mask tensor dimensions");
                }
                
                C = board_sizes[0];
                H = board_sizes[1];
                W = board_sizes[2];
                action_space = mask_sizes[0];
                
                dimensions_detected = true;
                
                std::cout << "[Inference Worker] Detected dimensions: C=" << C
                          << ", H=" << H << ", W=" << W
                          << ", A=" << action_space << std::endl;
            }
            
            torch::Tensor batch_boards =
                torch::zeros({actual_batch_size, C, H, W}, torch::kFloat32);
            torch::Tensor batch_masks =
                torch::zeros({actual_batch_size, action_space}, torch::kFloat32);
            
            for (int i = 0; i < actual_batch_size; ++i) {
                if (requests[i]->board.sizes()[0] != C ||
                    requests[i]->board.sizes()[1] != H ||
                    requests[i]->board.sizes()[2] != W) {
                    std::cerr << "[Inference Worker] ERROR: Mismatched board dimensions"
                              << std::endl;
                    throw std::runtime_error("Mismatched board dimensions");
                }
                
                batch_boards[i].copy_(requests[i]->board);
                batch_masks[i].copy_(requests[i]->mask);
            }
            
            torch::Tensor input_batch = batch_boards.to(device);
            torch::Tensor mask_batch = batch_masks.to(device);
            
            torch::Tensor policy_batch, value_batch;
            {
                torch::NoGradGuard no_grad;
                auto result = network->predict(input_batch, mask_batch);
                policy_batch = result.first;
                value_batch = result.second;
            }
            
            policy_batch = policy_batch.to(torch::kCPU);
            value_batch = value_batch.to(torch::kCPU);
            
            for (int i = 0; i < actual_batch_size; ++i) {
                requests[i]->result_promise.set_value({
                    policy_batch[i].clone().detach(),
                    value_batch[i].clone().detach()
                });
            }
            
            total_batches_processed++;
            total_requests_processed += actual_batch_size;
            
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
