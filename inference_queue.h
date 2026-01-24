#ifndef INFERENCE_QUEUE_H
#define INFERENCE_QUEUE_H

#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

enum class ModelID {
    MODEL_1 = 0,
    MODEL_2 = 1
};

class InferenceQueue {
public:
    struct InferenceRequest {
        torch::Tensor board;
        torch::Tensor mask;
        ModelID model_id;  // NEW: Tag to identify which model to use
        std::promise<std::pair<torch::Tensor, torch::Tensor>> result_promise;
        
        InferenceRequest(const torch::Tensor& b, const torch::Tensor& m, ModelID mid)
            : board(b), mask(m), model_id(mid) {}
    };
    
    InferenceQueue() : shutdown_(false) {}
    
    // Evaluate with specific model
    std::pair<torch::Tensor, torch::Tensor> evaluate_and_wait(
        const torch::Tensor& board, 
        const torch::Tensor& mask,
        ModelID model_id = ModelID::MODEL_1);
    
    // Get batch of requests for a specific model
    std::vector<std::shared_ptr<InferenceRequest>> get_batch(
        int batch_size, 
        int timeout_ms,
        ModelID model_id);
    
    // Get pending count for specific model
    size_t pending_count(ModelID model_id) const;
    
    // Get total pending count
    size_t pending_count() const;
    
    void shutdown();
    bool is_shutdown() const;
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<InferenceRequest>> request_queue_;
    bool shutdown_;
};

#endif // INFERENCE_QUEUE_H