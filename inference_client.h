#pragma once

#include "inference_queue_shm.h"
#include <torch/torch.h>
#include <memory>

/**
 * InferenceClient
 * 
 * High-level adapter for MCTS code.
 * Handles Torch tensor conversion and provides convenient APIs.
 * 
 * This is the layer that MCTS code interacts with.
 * It converts between Torch tensors and raw float arrays for IPC.
 */
class InferenceClient {
public:
    /**
     * Result structure
     */
    struct EvalResult {
        torch::Tensor policy;  // [POLICY_SIZE] - policy probabilities
        float value;           // scalar value estimate
    };
    
    /**
     * Job handle (for async use later)
     */
    struct EvalHandle {
        uint64_t job_id;
        
        EvalHandle(uint64_t id) : job_id(id) {}
    };
    
    /**
     * Constructor
     * @param shm_name: Shared memory name (e.g., "/mcts_jax_inference")
     */
    explicit InferenceClient(const std::string& shm_name);
    
    /**
     * Constructor with existing queue (for sharing between multiple clients)
     * @param queue: Shared pointer to inference queue
     */
    explicit InferenceClient(std::shared_ptr<SharedMemoryInferenceQueue> queue);
    
    /**
     * Destructor
     */
    ~InferenceClient() = default;
    
    // ========================================================================
    // BLOCKING API (Day-One)
    // ========================================================================
    
    /**
     * Evaluate board and wait for result (BLOCKING)
     * 
     * This is the drop-in replacement for old evaluate_and_wait()
     * 
     * @param board: Input tensor [C, H, W] or [1, C, H, W]
     * @param legal_mask: Legal moves [POLICY_SIZE] or [1, POLICY_SIZE]
     * @return EvalResult with (policy, value)
     * 
     * Throws std::runtime_error on timeout or error
     */
    EvalResult evaluate_and_wait(
        const torch::Tensor& board,
        const torch::Tensor& legal_mask);
    
    // ========================================================================
    // ASYNC API (Future Use)
    // ========================================================================
    
    /**
     * Submit evaluation request (NON-BLOCKING)
     * 
     * @param board: Input tensor [C, H, W] or [1, C, H, W]
     * @param legal_mask: Legal moves [POLICY_SIZE] or [1, POLICY_SIZE]
     * @return EvalHandle to check status and retrieve result
     */
    EvalHandle evaluate(
        const torch::Tensor& board,
        const torch::Tensor& legal_mask);
    
    /**
     * Check if evaluation is done
     * @param handle: Handle from evaluate()
     * @return true if result is ready
     */
    bool is_ready(const EvalHandle& handle);
    
    /**
     * Get result (blocking if not ready)
     * @param handle: Handle from evaluate()
     * @return EvalResult with (policy, value)
     * 
     * Throws std::runtime_error if job not found or timeout
     */
    EvalResult get(const EvalHandle& handle);
    
    // ========================================================================
    // STATUS
    // ========================================================================
    
    /**
     * Check if Python server is ready
     */
    bool is_server_ready() const;
    
    /**
     * Wait for server to be ready
     * @param timeout_ms: Maximum wait time in milliseconds
     * @return true if server became ready, false on timeout
     */
    bool wait_for_server(int timeout_ms = 30000);
    
    /**
     * Count pending requests
     */
    size_t pending_count() const;
    
    /**
     * Get statistics
     */
    SharedMemoryInferenceQueue::Stats get_stats() const;
    
    /**
     * Request shutdown
     */
    void request_shutdown();
    
private:
    std::shared_ptr<SharedMemoryInferenceQueue> queue_;
    
    // Conversion helpers
    void tensor_to_float_array(
        const torch::Tensor& tensor, 
        float* output, 
        size_t expected_size);
    
    torch::Tensor float_array_to_tensor(
        const float* data, 
        size_t size);
    
    // Validation
    void validate_board_tensor(const torch::Tensor& board);
    void validate_mask_tensor(const torch::Tensor& mask);
};