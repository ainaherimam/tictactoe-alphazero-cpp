#ifndef INFERENCE_WORKER_H
#define INFERENCE_WORKER_H

#include "inference_queue.h"
#include "alphaz_model.h"
#include <torch/torch.h>
#include <atomic>
#include <memory>
#include <vector>

/**
 * @brief Single-model inference worker for processing batched inference requests.
 *
 * This function runs in a thread to process inference requests from the queue
 * using a one NN model. Used for self-play
 *
 * @param queue Reference to the inference queue containing pending requests
 * @param network The AlphaZ neural network model
 * @param batch_size Maximum number of requests to process in a single batch
 * @param stop_flag Flag to signal worker shutdown
 */
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
    int batch_size,
    std::atomic<bool>& stop_flag);

/**
 * @brief Dual-model inference worker for processing requests for two different models (for evaluations)
 *
 * This function runs in a thread to process inference requests from the queue
 * using two different NN models. Used for model evaluation or comparison
 *
 * @param queue Reference to the inference queue containing pending requests
 * @param model1 The first AlphaZ neural network model
 * @param model2 The second AlphaZ neural network model
 * @param batch_size Maximum number of requests to process in a single batch per model
 * @param stop_flag Flag to signal worker shutdown
 */
void dual_inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> model1,
    std::shared_ptr<AlphaZModel> model2,
    int batch_size,
    std::atomic<bool>& stop_flag);

/**
 * @brief Helper function to process a batch of inference requests for a specific model.
 *
 * This function takes a batch of inference requests, runs them through the specified model,
 * and fulfills the promises with the resulting policy and value predictions.
 *
 * @param requests Vector of inference requests to process
 * @param model Shared pointer to the AlphaZ model to use for inference
 * @param device Device to run inference on
 * @param C Channels dimension 
 * @param H Height dimension 
 * @param W Width dimension
 * @param action_space Action space size
 * @param dimensions_detected Flag indicating if dimensions have been detected
 * @param model_id Identifier of the model being used (for verification)
 */
void process_batch(
    std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>>& requests,
    std::shared_ptr<AlphaZModel> model,
    torch::Device device,
    int& C, int& H, int& W, int& action_space,
    bool& dimensions_detected,
    ModelID model_id);

#endif // INFERENCE_WORKER_H