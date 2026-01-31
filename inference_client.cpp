#include "inference_client.h"
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iostream>

InferenceClient::InferenceClient(const std::string& shm_name) {
    queue_ = std::make_shared<SharedMemoryInferenceQueue>(shm_name);
}

InferenceClient::InferenceClient(
    std::shared_ptr<SharedMemoryInferenceQueue> queue) 
    : queue_(queue) {
    if (!queue_) {
        throw std::invalid_argument("Null queue pointer passed to InferenceClient");
    }
}

// ============================================================================
// BLOCKING API
// ============================================================================

InferenceClient::EvalResult InferenceClient::evaluate_and_wait(
    const torch::Tensor& board,
    const torch::Tensor& legal_mask) {
    
    // Validate inputs
    validate_board_tensor(board);
    validate_mask_tensor(legal_mask);
    
    // Flatten and convert tensors to float arrays
    float board_array[INPUT_SIZE];
    float mask_array[POLICY_SIZE];
    
    tensor_to_float_array(board, board_array, INPUT_SIZE);
    tensor_to_float_array(legal_mask, mask_array, POLICY_SIZE);
    
    // Submit to queue
    uint64_t job_id = queue_->submit(board_array, mask_array);
    
    // Wait for response
    float policy_array[POLICY_SIZE];
    float value;
    queue_->wait(job_id, policy_array, &value);
    
    // Convert back to tensors
    auto policy = float_array_to_tensor(policy_array, POLICY_SIZE);
    
    return {policy, value};
}

// ============================================================================
// ASYNC API
// ============================================================================

InferenceClient::EvalHandle InferenceClient::evaluate(
    const torch::Tensor& board,
    const torch::Tensor& legal_mask) {
    
    // Validate inputs
    validate_board_tensor(board);
    validate_mask_tensor(legal_mask);
    
    // Convert to float arrays
    float board_array[INPUT_SIZE];
    float mask_array[POLICY_SIZE];
    
    tensor_to_float_array(board, board_array, INPUT_SIZE);
    tensor_to_float_array(legal_mask, mask_array, POLICY_SIZE);
    
    // Submit to queue
    uint64_t job_id = queue_->submit(board_array, mask_array);
    
    return EvalHandle(job_id);
}

bool InferenceClient::is_ready(const EvalHandle& handle) {
    return queue_->is_done(handle.job_id);
}

InferenceClient::EvalResult InferenceClient::get(const EvalHandle& handle) {
    float policy_array[POLICY_SIZE];
    float value;
    
    // Wait for the job to complete
    queue_->wait(handle.job_id, policy_array, &value);
    
    // Convert to tensor
    auto policy = float_array_to_tensor(policy_array, POLICY_SIZE);
    
    return {policy, value};
}

// ============================================================================
// STATUS
// ============================================================================

bool InferenceClient::is_server_ready() const {
    return queue_->is_server_ready();
}

bool InferenceClient::wait_for_server(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);
    
    std::cout << "[InferenceClient] Waiting for Python server..." << std::flush;
    
    int dots = 0;
    while (!queue_->is_server_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > timeout) {
            std::cout << " TIMEOUT!" << std::endl;
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (++dots % 10 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    std::cout << " ✓ Connected!" << std::endl;
    return true;
}

size_t InferenceClient::pending_count() const {
    return queue_->count_pending();
}

SharedMemoryInferenceQueue::Stats InferenceClient::get_stats() const {
    return queue_->get_stats();
}

void InferenceClient::request_shutdown() {
    queue_->request_shutdown();
}

// ============================================================================
// HELPERS
// ============================================================================

void InferenceClient::tensor_to_float_array(
    const torch::Tensor& tensor,
    float* output,
    size_t expected_size) {
    
    // Flatten and convert to contiguous float tensor
    auto flat = tensor.flatten().to(torch::kFloat32).contiguous();
    
    if (static_cast<size_t>(flat.numel()) != expected_size) {
        throw std::runtime_error(
            "Tensor size mismatch: expected " + 
            std::to_string(expected_size) + 
            ", got " + std::to_string(flat.numel()));
    }
    
    // Copy to output array
    std::memcpy(output, flat.data_ptr<float>(), 
                expected_size * sizeof(float));
}

torch::Tensor InferenceClient::float_array_to_tensor(
    const float* data,
    size_t size) {
    
    // Create tensor from raw data (with copy)
    return torch::from_blob(
        const_cast<float*>(data),  // from_blob needs non-const
        {static_cast<long>(size)},
        torch::kFloat32
    ).clone();  // clone to own the data
}

void InferenceClient::validate_board_tensor(const torch::Tensor& board) {
    // Check that tensor has correct total size
    auto numel = board.numel();
    if (static_cast<size_t>(numel) != INPUT_SIZE) {
        throw std::runtime_error(
            "Board tensor has wrong size. Expected " + 
            std::to_string(INPUT_SIZE) + " elements (C=" + 
            std::to_string(INPUT_CHANNELS) + ", H=" + 
            std::to_string(BOARD_HEIGHT) + ", W=" + 
            std::to_string(BOARD_WIDTH) + "), got " + 
            std::to_string(numel));
    }
    
    // Expected shapes: [C, H, W] or [1, C, H, W]
    auto sizes = board.sizes();
    if (sizes.size() != 3 && sizes.size() != 4) {
        throw std::runtime_error(
            "Board tensor must be 3D [C,H,W] or 4D [1,C,H,W], got " + 
            std::to_string(sizes.size()) + "D");
    }
}

void InferenceClient::validate_mask_tensor(const torch::Tensor& mask) {
    // Check that tensor has correct total size
    auto numel = mask.numel();
    if (static_cast<size_t>(numel) != POLICY_SIZE) {
        throw std::runtime_error(
            "Mask tensor has wrong size. Expected " + 
            std::to_string(POLICY_SIZE) + " elements, got " + 
            std::to_string(numel));
    }
    
    // Expected shapes: [POLICY_SIZE] or [1, POLICY_SIZE]
    auto sizes = mask.sizes();
    if (sizes.size() != 1 && sizes.size() != 2) {
        throw std::runtime_error(
            "Mask tensor must be 1D or 2D, got " + 
            std::to_string(sizes.size()) + "D");
    }
}