#ifndef INFERENCE_QUEUE_H
#define INFERENCE_QUEUE_H

#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

/**
 * @enum ModelID
 * @brief Which neural network model to use for inference.
 */
enum class ModelID {
    MODEL_1 = 0, 
    MODEL_2 = 1 
};

/**
 * @class InferenceQueue
 * @brief Thread-safe queue for batching and processing neural network inference requests.
 *
 * This class manages concurrent inference requests for multiple models, organizing them
 * into separate queues for efficient batch processing.
 */
class InferenceQueue {
public:
    /**
     * @struct InferenceRequest
     * @brief Define a single inference request with its input data and result promise.
     */
    struct InferenceRequest {
        torch::Tensor board;                                              
        torch::Tensor mask;                                               
        ModelID model_id;                                                 
        std::promise<std::pair<torch::Tensor, torch::Tensor>> result_promise;  ///< Promise for policy and value results

        /**
         * @brief Constructs an inference request.
         * @param b Board state tensor
         * @param m Legal move mask tensor
         * @param mid Model identifier to use for inference
         */
        InferenceRequest(const torch::Tensor& b, const torch::Tensor& m, ModelID mid)
            : board(b), mask(m), model_id(mid) {}
    };

    /**
     * @brief Constructs a new InferenceQueue.
     */
    InferenceQueue() : shutdown_(false) {}

    /**
     * @brief Submits an inference request and waits for the result.
     * @param board Board state tensor to evaluate
     * @param mask Legal move mask tensor
     * @param model_id Model to use for inference (default: MODEL_1)
     * @return Pair of tensors containing policy and value predictions
     */
    std::pair<torch::Tensor, torch::Tensor> evaluate_and_wait(
        const torch::Tensor& board,
        const torch::Tensor& mask,
        ModelID model_id = ModelID::MODEL_1);

    /**
     * @brief Retrieves a batch of inference requests for a specific model.
     * @param batch_size Maximum number of requests to retrieve
     * @param timeout_ms Maximum time to wait for requests in milliseconds
     * @param model_id Model identifier to get requests for
     * @return Vector of inference requests for the specified model
     */
    std::vector<std::shared_ptr<InferenceRequest>> get_batch(
        int batch_size,
        int timeout_ms,
        ModelID model_id);

    /**
     * @brief Returns the number of pending requests for a given model.
     * @param model_id Model id to query
     * @return Number of pending requests for the model
     */
    size_t pending_count(ModelID model_id) const;

    /**
     * @brief Returns the total number of pending requests of all models.
     * @return Total number of pending requests
     */
    size_t pending_count() const;

    /**
     * @brief Graceful shutdown of the queue.
     */
    void shutdown();

    /**
     * @brief Checks if the queue is in shutdown state.
     * @return True if shutdown has been initiated, false otherwise
     */
    bool is_shutdown() const;

private:
    mutable std::mutex mutex_;                                       ///< Mutex for thread-safe access
    std::condition_variable cv_;                                     
    std::queue<std::shared_ptr<InferenceRequest>> queue_model1_;    
    std::queue<std::shared_ptr<InferenceRequest>> queue_model2_;    
    bool shutdown_;                                                  

    /**
     * @brief Helper function to get the queue for a specific model.
     * @param model_id Model identifier
     * @return Reference to the queue for the specified model
     */
    std::queue<std::shared_ptr<InferenceRequest>>& get_queue(ModelID model_id) {
        return (model_id == ModelID::MODEL_1) ? queue_model1_ : queue_model2_;
    }

    /**
     * @brief Helper function to get the const queue for a specific model.
     * @param model_id Model identifier
     * @return Const reference to the queue for the specified model
     */
    const std::queue<std::shared_ptr<InferenceRequest>>& get_queue(ModelID model_id) const {
        return (model_id == ModelID::MODEL_1) ? queue_model1_ : queue_model2_;
    }
};

#endif // INFERENCE_QUEUE_H