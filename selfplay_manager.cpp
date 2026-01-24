#include "selfplay_manager.h"
#include "game.h"
#include "mcts_agent_selfplay.h"
#include "logger.h"
#include <thread>
#include <iostream>
#include <chrono>
#include <exception>

// Forward declarations
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
    int batch_size,
    std::atomic<bool>& stop_flag);

void dual_inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> model1,
    std::shared_ptr<AlphaZModel> model2,
    int batch_size,
    std::atomic<bool>& stop_flag);

void SelfPlayManager::generate_training_data(
    int num_parallel_games,
    int games_per_worker,
    std::shared_ptr<AlphaZModel> network,
    GameDataset& dataset,
    int batch_size,
    int mcts_simulations,
    int board_size,
    double exploration_factor,
    float dirichlet_alpha,
    float dirichlet_epsilon) {
    
    std::cout << "\n==============================================\n";
    std::cout << "[SelfPlay] Starting Parallel Self-Play\n";
    std::cout << "==============================================\n";
    std::cout << "[SelfPlay] num_parallel_games = " << num_parallel_games << "\n";
    std::cout << "[SelfPlay] games_per_worker   = " << games_per_worker << "\n";
    std::cout << "[SelfPlay] total_games        = " << (num_parallel_games * games_per_worker) << "\n";
    std::cout << "[SelfPlay] batch_size         = " << batch_size << "\n";
    std::cout << "[SelfPlay] mcts_simulations   = " << mcts_simulations << "\n";
    std::cout << "[SelfPlay] board_size         = " << board_size << "\n";
    std::cout << "[SelfPlay] exploration_factor = " << exploration_factor << "\n";
    std::cout << "[SelfPlay] dirichlet_alpha    = " << dirichlet_alpha << "\n";
    std::cout << "[SelfPlay] dirichlet_epsilon  = " << dirichlet_epsilon << "\n";
    std::cout << "==============================================\n" << std::flush;
    
    if (!network) {
        throw std::runtime_error("[SelfPlay] ERROR: network is nullptr!");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    InferenceQueue inference_queue;
    std::atomic<int> games_completed{0};
    std::atomic<bool> stop_inference{false};
    
    // Launch single-model inference thread
    std::thread inference_thread;
    try {
        inference_thread = std::thread(
            inference_worker,
            std::ref(inference_queue),
            network,
            batch_size,
            std::ref(stop_inference));
    } catch (const std::exception& e) {
        std::cerr << "\n[SelfPlay] ERROR: Failed to launch inference thread: " 
                  << e.what() << std::endl;
        throw;
    }
    
    // Launch game worker threads
    std::vector<std::thread> game_threads;
    game_threads.reserve(num_parallel_games);
    
    for (int i = 0; i < num_parallel_games; ++i) {
        try {
            game_threads.emplace_back(
                &SelfPlayManager::game_worker,
                this,
                i,
                games_per_worker,
                &inference_queue,
                &dataset,
                board_size,
                mcts_simulations,
                exploration_factor,
                dirichlet_alpha,
                dirichlet_epsilon,
                &games_completed);
        } catch (const std::exception& e) {
            std::cerr << "\n[SelfPlay] ERROR: Failed to launch game worker " << i 
                      << ": " << e.what() << std::endl;
            
            stop_inference.store(true);
            inference_queue.shutdown();
            
            for (auto& thread : game_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            
            if (inference_thread.joinable()) {
                inference_thread.join();
            }
            
            throw;
        }
    }
    
    // Monitor progress
    int last_reported = 0;
    int total_games = num_parallel_games * games_per_worker;
    
    while (true) {
        int current = games_completed.load(std::memory_order_acquire);
        if (current >= total_games) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        current = games_completed.load(std::memory_order_acquire);
        size_t pending_inference = inference_queue.pending_count();
        
        if (current > last_reported) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            float games_per_sec = (elapsed_sec > 0) ? (current / (float)elapsed_sec) : 0.0f;
            
            std::cout << "\n[SelfPlay] Progress: " << current << "/" << total_games 
                      << " games (" << std::fixed << std::setprecision(1) 
                      << (100.0 * current / total_games) << "%), "
                      << std::fixed << std::setprecision(2) << games_per_sec << " games/sec, "
                      << "queue: " << pending_inference << " pending\n" << std::flush;
            last_reported = current;
        }
    }
    
    // Wait for all game threads
    for (auto& thread : game_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Signal and wait for inference thread
    stop_inference.store(true, std::memory_order_release);
    inference_queue.shutdown();
    
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    std::cout << "\n==============================================\n";
    std::cout << "[SelfPlay] Self-Play Complete\n";
    std::cout << "==============================================\n";
    std::cout << "[SelfPlay] Total games: " << total_games << "\n";
    std::cout << "[SelfPlay] Total time: " << total_duration << " seconds\n";
    std::cout << "[SelfPlay] Games per second: " 
              << ((total_duration > 0) ? (total_games / (float)total_duration) : 0.0f) << "\n";
    std::cout << "[SelfPlay] Training examples: " << dataset.actual_size() << "\n";
    std::cout << "==============================================\n\n" << std::flush;
}

EvaluationResults SelfPlayManager::generate_evaluation_games(
    int num_parallel_games,
    int games_per_worker,
    std::shared_ptr<AlphaZModel> model1,
    std::shared_ptr<AlphaZModel> model2,
    int batch_size,
    int mcts_simulations,
    int board_size,
    double exploration_factor,
    float dirichlet_alpha,
    float dirichlet_epsilon) {
    
    std::cout << "\n==============================================\n";
    std::cout << "[Evaluation] Starting Parallel Evaluation Games\n";
    std::cout << "==============================================\n";
    std::cout << "[Evaluation] num_parallel_games = " << num_parallel_games << "\n";
    std::cout << "[Evaluation] games_per_worker   = " << games_per_worker << "\n";
    std::cout << "[Evaluation] total_games        = " << (num_parallel_games * games_per_worker) << "\n";
    std::cout << "[Evaluation] batch_size         = " << batch_size << "\n";
    std::cout << "[Evaluation] mcts_simulations   = " << mcts_simulations << "\n";
    std::cout << "[Evaluation] exploration_factor = " << exploration_factor << "\n";
    std::cout << "[Evaluation] dirichlet_epsilon  = " << dirichlet_epsilon << " (should be 0.0)\n";
    std::cout << "[Evaluation] Using UNIFIED inference queue with dual-model worker\n";
    std::cout << "==============================================\n" << std::flush;
    
    // Validate model pointers
    if (!model1 || !model2) {
        throw std::runtime_error("[Evaluation] ERROR: model1 or model2 is nullptr!");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // UNIFIED INFERENCE QUEUE - handles both models
    InferenceQueue unified_queue;
    
    // Atomic counters and flags
    std::atomic<int> games_completed{0};
    std::atomic<int> model1_wins{0};
    std::atomic<int> model2_wins{0};
    std::atomic<int> draws{0};
    std::atomic<bool> stop_inference{false};
    
    // Launch SINGLE dual-model inference thread
    std::cout << "[Evaluation] Launching unified dual-model inference thread...\n" << std::flush;
    std::thread inference_thread(
        dual_inference_worker,
        std::ref(unified_queue),
        model1,
        model2,
        batch_size,
        std::ref(stop_inference));
    
    // Launch game worker threads
    std::vector<std::thread> game_threads;
    game_threads.reserve(num_parallel_games);
    
    std::cout << "[Evaluation] Launching " << num_parallel_games << " game workers...\n" << std::flush;
    for (int i = 0; i < num_parallel_games; ++i) {
        game_threads.emplace_back(
            &SelfPlayManager::evaluation_game_worker,
            this,
            i,
            games_per_worker,
            &unified_queue,  // Same queue for both models
            board_size,
            mcts_simulations,
            exploration_factor,
            dirichlet_alpha,
            dirichlet_epsilon,
            &games_completed,
            &model1_wins,
            &model2_wins,
            &draws);
    }
    
    // Monitor progress
    int last_reported = 0;
    int total_games = num_parallel_games * games_per_worker;
    
    while (games_completed.load(std::memory_order_acquire) < total_games) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        int current = games_completed.load(std::memory_order_acquire);
        if (current > last_reported) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            float games_per_sec = (elapsed_sec > 0) ? (current / (float)elapsed_sec) : 0.0f;
            
            size_t pending_m1 = unified_queue.pending_count(ModelID::MODEL_1);
            size_t pending_m2 = unified_queue.pending_count(ModelID::MODEL_2);
            
            std::cout << "[Evaluation] Progress: " << current << "/" << total_games 
                      << " games (" << std::fixed << std::setprecision(1) 
                      << (100.0 * current / total_games) << "%), "
                      << std::fixed << std::setprecision(2) << games_per_sec << " games/sec"
                      << " | M1: " << model1_wins.load()
                      << " M2: " << model2_wins.load()
                      << " Draws: " << draws.load()
                      << " | Queue M1:" << pending_m1 << " M2:" << pending_m2 << "\n" << std::flush;
            last_reported = current;
        }
    }
    
    // Wait for all game threads
    for (auto& thread : game_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Signal and wait for inference thread
    stop_inference.store(true, std::memory_order_release);
    unified_queue.shutdown();
    
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    // Prepare results
    EvaluationResults results;
    results.model1_wins = model1_wins.load();
    results.model2_wins = model2_wins.load();
    results.draws = draws.load();
    results.total_games = total_games;
    
    // Calculate win rates (scoring: win=1, draw=0.5, loss=0)
    float model1_score = results.model1_wins + (results.draws * 0.5f);
    float model2_score = results.model2_wins + (results.draws * 0.5f);
    
    results.model1_winrate = (model1_score / total_games) * 100.0f;
    results.model2_winrate = (model2_score / total_games) * 100.0f;
    
    std::cout << "\n==============================================\n";
    std::cout << "[Evaluation] Evaluation Complete\n";
    std::cout << "==============================================\n";
    std::cout << "[Evaluation] Total games: " << total_games << "\n";
    std::cout << "[Evaluation] Total time: " << total_duration << " seconds\n";
    std::cout << "[Evaluation] Model 1 wins: " << results.model1_wins << "\n";
    std::cout << "[Evaluation] Model 2 wins: " << results.model2_wins << "\n";
    std::cout << "[Evaluation] Draws: " << results.draws << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Evaluation] Model 1 win rate: " << results.model1_winrate << "%\n";
    std::cout << "[Evaluation] Model 2 win rate: " << results.model2_winrate << "%\n";
    std::cout << "==============================================\n\n" << std::flush;
    
    return results;
}

void SelfPlayManager::evaluation_game_worker(
    int worker_id,
    int num_games,
    InferenceQueue* queue,  // Single unified queue
    int board_size,
    int mcts_simulations,
    double exploration_factor,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    std::atomic<int>* games_completed,
    std::atomic<int>* model1_wins,
    std::atomic<int>* model2_wins,
    std::atomic<int>* draws) {
    
    std::cout << "[EvalWorker " << worker_id << "] Starting (will play " << num_games << " games)\n" << std::flush;
    
    // Validate pointers
    if (!queue || !games_completed || !model1_wins || !model2_wins || !draws) {
        std::cerr << "[EvalWorker " << worker_id << "] ERROR: nullptr received!" << std::endl;
        return;
    }
    
    // Dummy dataset for evaluation
    GameDataset dummy_dataset(100);
    
    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        try {
            // Create players with MODEL_ID tags
            auto player1 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                0.0f,  // Deterministic for evaluation
                dirichlet_alpha,
                dirichlet_epsilon,  // Should be 0.0 for evaluation
                queue,
                -1,
                false,
                ModelID::MODEL_1);  // Tag for model 1
            
            auto player2 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                0.0f,  // Deterministic for evaluation
                dirichlet_alpha,
                dirichlet_epsilon,  // Should be 0.0 for evaluation
                queue,
                -1,
                false,
                ModelID::MODEL_2);  // Tag for model 2
            
            // Create and play game (evaluation=true means don't save to dataset)
            Game game(board_size, std::move(player1), std::move(player2), 
                     dummy_dataset, true);
            
            Cell_state winner = game.play();
            
            // Update win counters
            if (winner == Cell_state::X) {
                model1_wins->fetch_add(1, std::memory_order_relaxed);
            } else if (winner == Cell_state::O) {
                model2_wins->fetch_add(1, std::memory_order_relaxed);
            } else {
                draws->fetch_add(1, std::memory_order_relaxed);
            }
            
            // Update completion counter
            games_completed->fetch_add(1, std::memory_order_release);
            
        } catch (const std::exception& e) {
            std::cerr << "[EvalWorker " << worker_id << "] ERROR in game " 
                      << (game_idx + 1) << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "[EvalWorker " << worker_id << "] Finished all games\n" << std::flush;
}

void SelfPlayManager::game_worker(
    int worker_id,
    int num_games,
    InferenceQueue* queue,
    GameDataset* dataset,
    int board_size,
    int mcts_simulations,
    double exploration_factor,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    std::atomic<int>* games_completed) {
    
    std::cout << "[Worker " << worker_id << "] Starting (will play " << num_games << " games)\n" << std::flush;
    
    if (!queue || !dataset || !games_completed) {
        std::cerr << "[Worker " << worker_id << "] ERROR: nullptr received!" << std::endl;
        return;
    }
    
    GameDataset local_dataset(num_games * 20);
    
    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        try {
            // For self-play, both players use MODEL_1 (same model)
            auto player1 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                1.0f,
                dirichlet_alpha,
                dirichlet_epsilon,
                queue,
                -1,
                false,
                ModelID::MODEL_1);

            
            auto player2 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                1.0f,
                dirichlet_alpha,
                dirichlet_epsilon,
                queue,
                -1,
                false,
                ModelID::MODEL_1);
            
            Game game(board_size, std::move(player1), std::move(player2), 
                     local_dataset, false);
            
            Cell_state winner = game.play();
            
            int completed = games_completed->fetch_add(1, std::memory_order_release) + 1;
            
            std::cout << "[Worker " << worker_id << "] Completed game " 
                      << (game_idx + 1) << "/" << num_games 
                      << ", winner: " << static_cast<int>(winner)
                      << " (total games: " << completed << ")\n" << std::flush;
            
        } catch (const std::exception& e) {
            std::cerr << "[Worker " << worker_id << "] ERROR in game " 
                      << (game_idx + 1) << ": " << e.what() << std::endl;
        }
    }
    
    try {
        dataset->merge(local_dataset);
    } catch (const std::exception& e) {
        std::cerr << "[Worker " << worker_id << "] ERROR merging dataset: " 
                  << e.what() << std::endl;
    }
    
    std::cout << "[Worker " << worker_id << "] Finished all games\n" << std::flush;
}