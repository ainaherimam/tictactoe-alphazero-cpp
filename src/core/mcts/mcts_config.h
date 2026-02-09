#ifndef MCTS_CONFIG_H
#define MCTS_CONFIG_H

#include <memory>
#include "core/utils/logger.h"
#include "inference/shared_memory/inference_queue_shm.h"
#include "inference/triton/triton_inference_client.h"

/**
 * @brief Configuration structure for MCTS players and agents
 *
 * Encapsulates all parameters needed to initialize MCTS-based players.
 */
struct Mcts_config {
    double exploration_factor;
    int number_iteration;
    LogLevel log_level;
    float temperature;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::shared_ptr<SharedMemoryInferenceQueue> queue;
    int max_depth;
    bool tree_reuse;
    uint32_t model_id;

    /**
     * @brief Construct MCTS configuration with default values
     */
    Mcts_config(double exploration_factor = 1.4,
                int number_iteration = 100,
                LogLevel log_level = LogLevel::NONE,
                float temperature = 0.0f,
                float dirichlet_alpha = 0.3f,
                float dirichlet_epsilon = 0.25f,
                std::shared_ptr<SharedMemoryInferenceQueue> queue = nullptr,
                int max_depth = 100,
                bool tree_reuse = false,
                uint32_t model_id = 0)
        : exploration_factor(exploration_factor),
          number_iteration(number_iteration),
          log_level(log_level),
          temperature(temperature),
          dirichlet_alpha(dirichlet_alpha),
          dirichlet_epsilon(dirichlet_epsilon),
          queue(queue),
          max_depth(max_depth),
          tree_reuse(tree_reuse),
          model_id(model_id) {}
};

/**
 * @brief Configuration structure for MCTS players and agents using Triton inference
 *
 * Encapsulates all parameters needed to initialize Triton-based MCTS players.
 */
struct Mcts_triton_config {
    double exploration_factor;
    int number_iteration;
    LogLevel log_level;
    float temperature;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    InferenceClient* client;
    int max_depth;
    bool tree_reuse;
    uint32_t model_id;

    /**
     * @brief Construct MCTS Triton configuration with default values
     */
    Mcts_triton_config(double exploration_factor = 1.4,
                       int number_iteration = 100,
                       LogLevel log_level = LogLevel::NONE,
                       float temperature = 0.0f,
                       float dirichlet_alpha = 0.3f,
                       float dirichlet_epsilon = 0.25f,
                       InferenceClient* client = nullptr,
                       int max_depth = 100,
                       bool tree_reuse = false,
                       uint32_t model_id = 0)
        : exploration_factor(exploration_factor),
          number_iteration(number_iteration),
          log_level(log_level),
          temperature(temperature),
          dirichlet_alpha(dirichlet_alpha),
          dirichlet_epsilon(dirichlet_epsilon),
          client(client),
          max_depth(max_depth),
          tree_reuse(tree_reuse),
          model_id(model_id) {}
};

#endif