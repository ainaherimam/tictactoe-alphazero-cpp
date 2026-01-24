#ifndef INFERENCE_WORKER_H
#define INFERENCE_WORKER_H

#include "inference_queue.h"
#include "alphaz_model.h"
#include <torch/torch.h>
#include <atomic>
#include <memory>
#include <vector>

// Single-model inference worker (for self-play)
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
    int batch_size,
    std::atomic<bool>& stop_flag);

// Dual-model inference worker (for evaluation)
void dual_inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> model1,
    std::shared_ptr<AlphaZModel> model2,
    int batch_size,
    std::atomic<bool>& stop_flag);

// Helper function to process a batch for a specific model
void process_batch(
    std::vector<std::shared_ptr<InferenceQueue::InferenceRequest>>& requests,
    std::shared_ptr<AlphaZModel> model,
    torch::Device device,
    int& C, int& H, int& W, int& action_space,
    bool& dimensions_detected,
    ModelID model_id);

#endif // INFERENCE_WORKER_H