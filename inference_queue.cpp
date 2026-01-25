#include "inference_queue.h"
#include <chrono>
#include <iostream>

std::pair<torch::Tensor, torch::Tensor> InferenceQueue::evaluate_and_wait(
    const torch::Tensor& board, 
    const torch::Tensor& mask,
    ModelID model_id) {
    
    // Check if shutdown before creating request
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("InferenceQueue is shut down - cannot submit new requests");
        }
    }
    
    // Create request with model ID tag
    auto request = std::make_shared<InferenceRequest>(
        board.clone(), 
        mask.clone(), 
        model_id);
    auto future = request->result_promise.get_future();
    
    // Add request to appropriate queue
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("InferenceQueue is shut down - cannot submit new requests");
        }
        get_queue(model_id).push(request);
    }
    
    // Notify inference thread
    cv_.notify_one();
    
    // Block until result is ready
    try {
        return future.get();
    } catch (const std::exception& e) {
        std::cerr << "[InferenceQueue] Exception while waiting for result: " 
                  << e.what() << std::endl;
        throw;
    }
}

std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>> 
InferenceQueue::get_batch(int batch_size, int timeout_ms, ModelID model_id) {
    
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    batch.reserve(batch_size);
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    auto deadline = std::chrono::steady_clock::now() + 
                   std::chrono::milliseconds(timeout_ms);
    
    auto& queue = get_queue(model_id);
    
    // Wait for at least one request for this model or shutdown
    while (queue.empty() && !shutdown_) {
        if (timeout_ms == 0) {
            // Return immediately if queue empty
            return batch;
        }
        
        auto status = cv_.wait_until(lock, deadline);
        if (status == std::cv_status::timeout) {
            return batch;  // Timeout - return empty batch
        }
    }
    
    // If shutdown and no requests, return empty batch
    if (shutdown_ && queue.empty()) {
        return batch;
    }
    
    // Collect requests from the specific model's queue
    while (!queue.empty() && batch.size() < static_cast<size_t>(batch_size)) {
        batch.push_back(queue.front());
        queue.pop();
    }
    
    return batch;
}

size_t InferenceQueue::pending_count(ModelID model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_queue(model_id).size();
}

size_t InferenceQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_model1_.size() + queue_model2_.size();
}

void InferenceQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

bool InferenceQueue::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}