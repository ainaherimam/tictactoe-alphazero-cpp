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
    
    // Add request to queue
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            throw std::runtime_error("InferenceQueue is shut down - cannot submit new requests");
        }
        request_queue_.push(request);
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
    
    // Wait for at least one request for this model or shutdown
    while (true) {
        // Check if there's any request for our model
        bool found = false;
        std::queue<std::shared_ptr<InferenceRequest>> temp_queue;
        
        while (!request_queue_.empty()) {
            auto req = request_queue_.front();
            request_queue_.pop();
            
            if (req->model_id == model_id) {
                found = true;
                request_queue_.push(req);  // Put it back
                break;
            } else {
                temp_queue.push(req);  // Save for later
            }
        }
        
        // Restore non-matching requests
        while (!temp_queue.empty()) {
            request_queue_.push(temp_queue.front());
            temp_queue.pop();
        }
        
        if (found || shutdown_) {
            break;
        }
        
        auto status = cv_.wait_until(lock, deadline);
        if (status == std::cv_status::timeout) {
            return batch;  // Timeout - return empty batch
        }
    }
    
    // If shutdown and no requests, return empty batch
    if (shutdown_ && request_queue_.empty()) {
        return batch;
    }
    
    // Collect requests for this specific model
    std::queue<std::shared_ptr<InferenceRequest>> temp_queue;
    
    while (!request_queue_.empty() && batch.size() < static_cast<size_t>(batch_size)) {
        auto req = request_queue_.front();
        request_queue_.pop();
        
        if (req->model_id == model_id) {
            batch.push_back(req);
        } else {
            temp_queue.push(req);  // Different model, save for later
        }
    }
    
    // Put back requests for other models
    while (!temp_queue.empty()) {
        request_queue_.push(temp_queue.front());
        temp_queue.pop();
    }
    
    return batch;
}

size_t InferenceQueue::pending_count(ModelID model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = 0;
    std::queue<std::shared_ptr<InferenceRequest>> temp_queue = request_queue_;
    
    while (!temp_queue.empty()) {
        if (temp_queue.front()->model_id == model_id) {
            count++;
        }
        temp_queue.pop();
    }
    
    return count;
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
    cv_.notify_all();
}

bool InferenceQueue::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}