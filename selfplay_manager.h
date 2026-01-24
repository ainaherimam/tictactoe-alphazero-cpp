#ifndef SELFPLAY_MANAGER_H
#define SELFPLAY_MANAGER_H

#include "alphaz_model.h"
#include "game_dataset.h"
#include "inference_queue.h"
#include <memory>
#include <atomic>
#include <vector>

struct EvaluationResults {
    int model1_wins;
    int model2_wins;
    int draws;
    int total_games;
    float model1_winrate;
    float model2_winrate;
};


/**
 * Manages parallel self-play training with batched inference.
 * Spawns multiple game threads and a single inference thread.
 */
class SelfPlayManager {
public:
    SelfPlayManager() = default;
    ~SelfPlayManager() = default;

    /**
     * Generate training data using parallel self-play.
     * 
     * @param num_parallel_games Number of games to run in parallel (e.g., 16-32)
     * @param games_per_worker Number of games each worker should play
     * @param network Shared neural network for inference
     * @param dataset Dataset to store training examples
     * @param batch_size Batch size for GPU inference (e.g., 16-32)
     * @param mcts_simulations Number of MCTS simulations per move
     * @param board_size Size of the game board
     * @param exploration_factor PUCT exploration constant
     * @param dirichlet_alpha Dirichlet noise alpha parameter
     * @param dirichlet_epsilon Dirichlet noise mixing weight
     */
    void generate_training_data(
        int num_parallel_games,
        int games_per_worker,
        std::shared_ptr<AlphaZModel> network,
        GameDataset& dataset,
        int batch_size = 32,
        int mcts_simulations = 200,
        int board_size = 4,
        double exploration_factor = 1.0,
        float dirichlet_alpha = 0.3f,
        float dirichlet_epsilon = 0.25f);

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
    /**
     * Worker function for a single game thread.
     * Plays multiple games and collects training data.
     */
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
    
    void evaluation_game_worker(
        int worker_id,
        int num_games,
        InferenceQueue* queue1,
        InferenceQueue* queue2,
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