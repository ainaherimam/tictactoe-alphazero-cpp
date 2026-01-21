#pragma once

#include <torch/torch.h>
#include <vector>
#include <mutex>
#include <utility>

/**
 * Thread-safe pool of pre-allocated tensors for efficient reuse.
 * Eliminates allocation overhead during MCTS inference.
 */
class TensorPool {
private:
    struct PoolEntry {
        torch::Tensor board_tensor;
        torch::Tensor mask_tensor;
        bool in_use;
        
        PoolEntry(torch::Tensor b, torch::Tensor m) 
            : board_tensor(std::move(b)), mask_tensor(std::move(m)), in_use(false) {}
    };
    
    std::vector<PoolEntry> pool;
    mutable std::mutex pool_mutex;
    torch::Device device;
    
    // Tensor dimensions
    const int board_channels;
    const int board_height;
    const int board_width;
    const int mask_size;
    
public:
    /**
     * Create a pool of pre-allocated tensors.
     * @param dev Device to allocate tensors on (CPU or CUDA)
     * @param pool_size Number of tensor pairs to pre-allocate
     * @param channels Board tensor channels (default: 11)
     * @param height Board height (default: 5)
     * @param width Board width (default: 9)
     * @param mask_sz Mask size (default: 1800)
     */
    TensorPool(torch::Device dev, 
               size_t pool_size = 32,
               int channels = 3, 
               int height = 4, 
               int width = 4,
               int mask_sz = 16);
    
    /**
     * Acquire a pair of tensors from the pool.
     * Returns pre-allocated GPU tensors (zeroed out).
     * If pool is exhausted, creates new tensors (should be rare).
     */
    std::pair<torch::Tensor, torch::Tensor> acquire();
    
    /**
     * Release tensors back to the pool for reuse.
     */
    void release(const torch::Tensor& board_tensor, const torch::Tensor& mask_tensor);
    
    /**
     * Get current pool statistics (for debugging/monitoring).
     */
    struct PoolStats {
        size_t total_size;
        size_t in_use_count;
        size_t available_count;
    };
    PoolStats get_stats() const;
};