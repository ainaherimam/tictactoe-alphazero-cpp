#ifndef INFERENCE_QUEUE_H
#define INFERENCE_QUEUE_H

#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <vector>

/**
 * Thread-safe queue for batched neural network inference requests.
 * Game threads submit requests and block until the inference thread processes them.
 */
class InferenceQueue {
public:
    struct InferenceRequest {
        torch::Tensor board;           // Board state tensor
        torch::Tensor mask;            // Legal move mask
        std::promise<std::pair<torch::Tensor, torch::Tensor>> result_promise;
        
        InferenceRequest(torch::Tensor b, torch::Tensor m)
            : board(std::move(b)), mask(std::move(m)) {}
    };

    InferenceQueue() = default;
    ~InferenceQueue() = default;

    // Non-copyable, non-movable
    InferenceQueue(const InferenceQueue&) = delete;
    InferenceQueue& operator=(const InferenceQueue&) = delete;

    /**
     * Submit an inference request and wait for the result.
     * This method blocks until the inference thread processes the request.
     * 
     * @param board Board state tensor (shape: [C, H, W])
     * @param mask Legal move mask tensor
     * @return Pair of (policy, value) from neural network
     */
    std::pair<torch::Tensor, torch::Tensor> evaluate_and_wait(
        const torch::Tensor& board, 
        const torch::Tensor& mask);

    /**
     * Get a batch of requests for processing (called by inference thread).
     * Waits for batch_size requests or timeout_ms milliseconds.
     * 
     * @param batch_size Desired batch size
     * @param timeout_ms Maximum time to wait for a full batch
     * @return Vector of requests (may be smaller than batch_size)
     */
    std::vector<std::shared_ptr<InferenceRequest>> get_batch(
        int batch_size, 
        int timeout_ms);

    /**
     * Get the current number of pending requests.
     */
    size_t pending_count() const;

    /**
     * Signal that no more requests will be submitted.
     * The inference thread should process remaining requests and exit.
     */
    void shutdown();

    /**
     * Check if shutdown has been requested.
     */
    bool is_shutdown() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<InferenceRequest>> request_queue_;
    bool shutdown_ = false;
};

#endif // INFERENCE_QUEUE_H