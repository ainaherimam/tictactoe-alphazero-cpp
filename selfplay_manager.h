#ifndef SELFPLAY_MANAGER_H
#define SELFPLAY_MANAGER_H

#include "alphaz_model.h"
#include "game_dataset.h"
#include "inference_queue.h"
#include <memory>
#include <atomic>

struct EvaluationResults {
    int model1_wins;
    int model2_wins;
    int draws;
    int total_games;
    float model1_winrate;
    float model2_winrate;
};

class SelfPlayManager {
public:
    SelfPlayManager() = default;
    ~SelfPlayManager() = default;
    
    // Generate training data through self-play
    void generate_training_data(
        int num_parallel_games,
        int games_per_worker,
        std::shared_ptr<AlphaZModel> network,
        GameDataset& dataset,
        int batch_size,
        int mcts_simulations,
        int board_size,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon);
    
    // Evaluate two models against each other
    EvaluationResults generate_evaluation_games(
        int num_parallel_games,
        int games_per_worker,
        std::shared_ptr<AlphaZModel> model1,
        std::shared_ptr<AlphaZModel> model2,
        int batch_size,
        int mcts_simulations,
        int board_size,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon);
    
private:
    // Worker for self-play games
    void game_worker(
        int worker_id,
        int num_games,
        InferenceQueue* queue,
        GameDataset* dataset,
        int board_size,
        int mcts_simulations,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        std::atomic<int>* games_completed);
    
    // Worker for evaluation games (UPDATED: single unified queue)
    void evaluation_game_worker(
        int worker_id,
        int num_games,
        InferenceQueue* queue,  // Single unified queue for both models
        int board_size,
        int mcts_simulations,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        std::atomic<int>* games_completed,
        std::atomic<int>* model1_wins,
        std::atomic<int>* model2_wins,
        std::atomic<int>* draws);
};

#endif // SELFPLAY_MANAGER_H