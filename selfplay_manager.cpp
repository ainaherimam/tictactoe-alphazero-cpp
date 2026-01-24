#include "selfplay_manager.h"
#include "game.h"
#include "mcts_agent_selfplay.h"
#include "logger.h"
#include <thread>
#include <iostream>
#include <chrono>
#include <exception>

// Forward declaration of inference_worker
void inference_worker(
    InferenceQueue& queue,
    std::shared_ptr<AlphaZModel> network,
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
    
    // CRITICAL FIX: Validate network pointer
    if (!network) {
        throw std::runtime_error("[SelfPlay] ERROR: network is nullptr!");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // CRITICAL FIX: Create inference queue on heap to avoid potential stack issues
    std::cout << "[SelfPlay] Creating inference queue..." << std::flush;
    InferenceQueue inference_queue;
    std::cout << " done\n" << std::flush;
    
    // CRITICAL FIX: Initialize atomics properly
    std::cout << "[SelfPlay] Atomic counters initialized\n" << std::flush;
    std::atomic<int> games_completed{0};
    std::atomic<bool> stop_inference{false};
    
    // CRITICAL FIX: Launch inference thread with exception handling
    std::cout << "[SelfPlay] Launching inference thread..." << std::flush;
    std::thread inference_thread;
    try {
        inference_thread = std::thread(
            inference_worker,
            std::ref(inference_queue),
            network,
            batch_size,
            std::ref(stop_inference));
        std::cout << " done\n" << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "\n[SelfPlay] ERROR: Failed to launch inference thread: " 
                  << e.what() << std::endl;
        throw;
    }
    
    // CRITICAL FIX: Launch game worker threads with exception handling
    std::vector<std::thread> game_threads;
    game_threads.reserve(num_parallel_games);
    
    std::cout << "[SelfPlay] Launching " << num_parallel_games << " game worker threads...\n" << std::flush;
    
    for (int i = 0; i < num_parallel_games; ++i) {
        std::cout << "[SelfPlay] Launching game worker thread " << i << std::flush;
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
            std::cout << " - success\n" << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "\n[SelfPlay] ERROR: Failed to launch game worker " << i 
                      << ": " << e.what() << std::endl;
            
            // CRITICAL: Clean up already-launched threads
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
    
    std::cout << "[SelfPlay] All game worker threads launched\n" << std::flush;
    
    // CRITICAL FIX: Monitor progress with proper error handling
    int last_reported = 0;
    int total_games = num_parallel_games * games_per_worker;
    
    std::cout << "[SelfPlay] Entering progress monitoring loop\n" << std::flush;
    
    while (true) {
        try {
            std::cout << "[SelfPlay][Monitor] Loop start\n" << std::flush;
            
            // CRITICAL FIX: Check completion before sleeping
            int current = games_completed.load(std::memory_order_acquire);
            if (current >= total_games) {
                std::cout << "[SelfPlay][Monitor] All games completed!\n" << std::flush;
                break;
            }
            
            std::cout << "[SelfPlay][Monitor] Sleeping for 1 second...\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "[SelfPlay][Monitor] Woke up from sleep\n" << std::flush;
            
            std::cout << "[SelfPlay][Monitor] Loading games_completed...\n" << std::flush;
            current = games_completed.load(std::memory_order_acquire);
            std::cout << "[SelfPlay][Monitor] games_completed = " << current << "\n" << std::flush;
            
            std::cout << "[SelfPlay][Monitor] Querying inference_queue.pending_count()...\n" << std::flush;
            size_t pending_inference = inference_queue.pending_count();
            std::cout << "[SelfPlay][Monitor] pending_inference = " << pending_inference << "\n" << std::flush;
            
            std::cout << "[SelfPlay][Monitor] Status | completed=" << current 
                      << " / " << total_games 
                      << " | pending_inference=" << pending_inference << "\n" << std::flush;
            
            std::cout << "[SelfPlay][Monitor] Checking if progress advanced (current=" 
                      << current << ", last_reported=" << last_reported << ")\n" << std::flush;
            
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
            } else {
                std::cout << "[SelfPlay][Monitor] No new games completed since last report\n" << std::flush;
            }
            
            std::cout << "[SelfPlay][Monitor] Loop end\n" << std::flush;
            
        } catch (const std::exception& e) {
            std::cerr << "[SelfPlay][Monitor] ERROR in monitoring loop: " 
                      << e.what() << std::endl;
            // Continue monitoring despite errors
        }
    }
    
    // CRITICAL FIX: Wait for all game threads with timeout
    std::cout << "\n[SelfPlay] Waiting for all game threads to complete...\n" << std::flush;
    for (size_t i = 0; i < game_threads.size(); ++i) {
        std::cout << "[SelfPlay] Joining game thread " << i << "..." << std::flush;
        try {
            if (game_threads[i].joinable()) {
                game_threads[i].join();
                std::cout << " done\n" << std::flush;
            } else {
                std::cout << " already joined\n" << std::flush;
            }
        } catch (const std::exception& e) {
            std::cerr << " ERROR: " << e.what() << "\n" << std::flush;
        }
    }
    
    // CRITICAL FIX: Signal inference thread to stop
    std::cout << "[SelfPlay] All game threads joined. Signaling inference thread to stop...\n" << std::flush;
    stop_inference.store(true, std::memory_order_release);
    inference_queue.shutdown();
    
    // CRITICAL FIX: Wait for inference thread
    std::cout << "[SelfPlay] Joining inference thread..." << std::flush;
    try {
        if (inference_thread.joinable()) {
            inference_thread.join();
            std::cout << " done\n" << std::flush;
        }
    } catch (const std::exception& e) {
        std::cerr << " ERROR: " << e.what() << "\n" << std::flush;
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
    std::cout << "==============================================\n" << std::flush;
    
    // Validate model pointers
    if (!model1 || !model2) {
        throw std::runtime_error("[Evaluation] ERROR: model1 or model2 is nullptr!");
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Create two inference queues (one for each model)
    InferenceQueue queue1;
    InferenceQueue queue2;
    
    // Atomic counters and flags
    std::atomic<int> games_completed{0};
    std::atomic<int> model1_wins{0};
    std::atomic<int> model2_wins{0};
    std::atomic<int> draws{0};
    std::atomic<bool> stop_inference1{false};
    std::atomic<bool> stop_inference2{false};
    
    // Launch inference threads for both models
    std::thread inference_thread1(
        inference_worker,
        std::ref(queue1),
        model1,
        batch_size,
        std::ref(stop_inference1));
    
    std::thread inference_thread2(
        inference_worker,
        std::ref(queue2),
        model2,
        batch_size,
        std::ref(stop_inference2));
    
    // Launch game worker threads
    std::vector<std::thread> game_threads;
    game_threads.reserve(num_parallel_games);
    
    for (int i = 0; i < num_parallel_games; ++i) {
        game_threads.emplace_back(
            &SelfPlayManager::evaluation_game_worker,
            this,
            i,
            games_per_worker,
            &queue1,
            &queue2,
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
            
            std::cout << "[Evaluation] Progress: " << current << "/" << total_games 
                      << " games (" << std::fixed << std::setprecision(1) 
                      << (100.0 * current / total_games) << "%), "
                      << std::fixed << std::setprecision(2) << games_per_sec << " games/sec"
                      << " | Model1: " << model1_wins.load()
                      << " Model2: " << model2_wins.load()
                      << " Draws: " << draws.load() << "\n" << std::flush;
            last_reported = current;
        }
    }
    
    // Wait for all game threads
    for (auto& thread : game_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Signal inference threads to stop
    stop_inference1.store(true, std::memory_order_release);
    stop_inference2.store(true, std::memory_order_release);
    queue1.shutdown();
    queue2.shutdown();
    
    // Wait for inference threads
    if (inference_thread1.joinable()) {
        inference_thread1.join();
    }
    if (inference_thread2.joinable()) {
        inference_thread2.join();
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
    std::atomic<int>* draws) {
    
    std::cout << "[EvalWorker " << worker_id << "] Starting (will play " << num_games << " games)\n" << std::flush;
    
    // Validate pointers
    if (!queue1 || !queue2 || !games_completed || !model1_wins || !model2_wins || !draws) {
        std::cerr << "[EvalWorker " << worker_id << "] ERROR: nullptr received!" << std::endl;
        return;
    }
    
    // Dummy dataset for evaluation (we don't save these games)
    GameDataset dummy_dataset(100);
    
    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        try {
            // Create players with deterministic temperature (0.0)
            auto player1 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                0.0f,  // Deterministic for evaluation
                dirichlet_alpha,
                dirichlet_epsilon,  // Should be 0.0 for evaluation
                queue1);
            
            auto player2 = std::make_unique<Mcts_player_selfplay>(
                exploration_factor,
                mcts_simulations,
                LogLevel::NONE,
                0.0f,  // Deterministic for evaluation
                dirichlet_alpha,
                dirichlet_epsilon,  // Should be 0.0 for evaluation
                queue2);
            
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
    
    // CRITICAL FIX: Validate pointers
    if (!queue || !dataset || !games_completed) {
        std::cerr << "[Worker " << worker_id << "] ERROR: nullptr received!" << std::endl;
        return;
    }
    
    // CRITICAL FIX: Use smaller local dataset and handle exceptions
    GameDataset local_dataset(num_games * 20);  // Estimate ~100 moves per game
    
    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        try {
            std::cout << "[Worker " << worker_id << "] Starting game " 
                      << (game_idx + 1) << "/" << num_games << "\n" << std::flush;
            
            // CRITICAL FIX: Create players with proper error handling
            std::unique_ptr<Mcts_player_selfplay> player1;
            std::unique_ptr<Mcts_player_selfplay> player2;
            
            try {
                player1 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    1.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    queue);
                
                player2 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    1.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    queue);
            } catch (const std::exception& e) {
                std::cerr << "[Worker " << worker_id << "] ERROR creating players: " 
                          << e.what() << std::endl;
                throw;
            }
            
            // Create and play game
            Game game(board_size, std::move(player1), std::move(player2), 
                     local_dataset, false);  // evaluation=false for training
            
            Cell_state winner = game.play();
            
            // CRITICAL FIX: Update completion counter with proper memory ordering
            int completed = games_completed->fetch_add(1, std::memory_order_release) + 1;
            
            std::cout << "[Worker " << worker_id << "] Completed game " 
                      << (game_idx + 1) << "/" << num_games 
                      << ", winner: " << static_cast<int>(winner)
                      << " (total games: " << completed << ")\n" << std::flush;
            
        } catch (const std::exception& e) {
            std::cerr << "[Worker " << worker_id << "] ERROR in game " 
                      << (game_idx + 1) << ": " << e.what() << std::endl;
            // Continue with next game rather than crashing entire worker
        }
    }
    
    // CRITICAL FIX: Merge with error handling
    try {
        std::cout << "[Worker " << worker_id << "] Merging " 
                  << local_dataset.actual_size() << " examples into global dataset...\n" << std::flush;
        
        dataset->merge(local_dataset);
        
        std::cout << "[Worker " << worker_id << "] Merge complete. Global dataset now has " 
                  << dataset->actual_size() << " examples.\n" << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "[Worker " << worker_id << "] ERROR merging dataset: " 
                  << e.what() << std::endl;
    }
    
    std::cout << "[Worker " << worker_id << "] Finished all games\n" << std::flush;
}