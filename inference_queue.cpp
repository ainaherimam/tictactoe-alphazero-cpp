#include "inference_queue.h"
#include <chrono>
#include <iostream>

std::pair<torch::Tensor, torch::Tensor> InferenceQueue::evaluate_and_wait(
    const torch::Tensor& board, 
    const torch::Tensor& mask) {
    
    // CRITICAL FIX: Check if shutdown before creating request
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("InferenceQueue is shut down - cannot submit new requests");
        }
    }
    
    // Create request with promise/future pair
    // SAFETY: Use make_shared to ensure proper memory management
    auto request = std::make_shared<InferenceRequest>(board.clone(), mask.clone());
    auto future = request->result_promise.get_future();
    
    // Add request to queue
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Double-check shutdown status while holding lock
        if (shutdown_) {
            throw std::runtime_error("InferenceQueue is shut down - cannot submit new requests");
        }
        request_queue_.push(request);
    }
    
    // Notify inference thread that a request is available
    cv_.notify_one();
    
    // CRITICAL: Block until result is ready
    // This may throw if inference thread sets an exception
    try {
        return future.get();
    } catch (const std::exception& e) {
        std::cerr << "[InferenceQueue] Exception while waiting for result: " << e.what() << std::endl;
        throw;
    }
}

std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>> 
InferenceQueue::get_batch(int batch_size, int timeout_ms) {
    
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    batch.reserve(batch_size);
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // CRITICAL FIX: Use proper timeout handling
    auto deadline = std::chrono::steady_clock::now() + 
                   std::chrono::milliseconds(timeout_ms);
    
    // Wait for at least one request or shutdown
    while (request_queue_.empty() && !shutdown_) {
        auto status = cv_.wait_until(lock, deadline);
        if (status == std::cv_status::timeout) {
            // Timeout - return empty batch
            return batch;
        }
    }
    
    // If shutdown and no requests, return empty batch
    if (shutdown_ && request_queue_.empty()) {
        return batch;
    }
    
    // CRITICAL FIX: Collect all available requests up to batch_size
    // Don't wait for more if we already have some
    while (!request_queue_.empty() && batch.size() < static_cast<size_t>(batch_size)) {
        batch.push_back(request_queue_.front());
        request_queue_.pop();
    }
    
    return batch;
}

size_t InferenceQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_queue_.size();
}

void InferenceQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    // CRITICAL: Notify all waiting threads
    cv_.notify_all();
}

bool InferenceQueue::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}