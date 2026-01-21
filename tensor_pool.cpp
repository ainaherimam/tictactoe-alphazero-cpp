#include "tensor_pool.h"
#include <iostream>

TensorPool::TensorPool(torch::Device dev, 
                       size_t pool_size,
                       int channels, 
                       int height, 
                       int width,
                       int mask_sz)
    : device(dev),
      board_channels(channels),
      board_height(height),
      board_width(width),
      mask_size(mask_sz)
{
    pool.reserve(pool_size);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);
    
    // std::cout << "Initializing TensorPool with " << pool_size 
    //           << " tensor pairs on device: " 
    //           << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    for (size_t i = 0; i < pool_size; ++i) {
        torch::Tensor board_tensor = torch::zeros(
            {board_channels, board_height, board_width}, options);
        torch::Tensor mask_tensor = torch::zeros({mask_size}, options);
        
        pool.emplace_back(std::move(board_tensor), std::move(mask_tensor));
    }
    
    // std::cout << "TensorPool initialized successfully" << std::endl;
}

std::pair<torch::Tensor, torch::Tensor> TensorPool::acquire() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Find an available entry
    for (auto& entry : pool) {
        if (!entry.in_use) {
            entry.in_use = true;
            // Zero out tensors for reuse
            entry.board_tensor.zero_();
            entry.mask_tensor.zero_();
            return {entry.board_tensor, entry.mask_tensor};
        }
    }
    
    // Pool exhausted - create new tensors (should be rare with proper sizing)
    std::cerr << "WARNING: TensorPool exhausted, creating new tensors. "
              << "Consider increasing pool size." << std::endl;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);
    
    return {
        torch::zeros({board_channels, board_height, board_width}, options),
        torch::zeros({mask_size}, options)
    };
}

void TensorPool::release(const torch::Tensor& board_tensor, 
                        const torch::Tensor& mask_tensor) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Find matching entry and mark as available
    for (auto& entry : pool) {
        if (entry.board_tensor.data_ptr() == board_tensor.data_ptr()) {
            entry.in_use = false;
            return;
        }
    }
    
    // Tensor not from pool (was dynamically allocated due to exhaustion)
    // Nothing to do - will be garbage collected
}

TensorPool::PoolStats TensorPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    size_t in_use = 0;
    for (const auto& entry : pool) {
        if (entry.in_use) ++in_use;
    }
    
    return {
        pool.size(),
        in_use,
        pool.size() - in_use
    };
}
