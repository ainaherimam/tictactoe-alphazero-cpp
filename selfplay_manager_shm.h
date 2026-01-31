#pragma once

#include "inference_queue_shm.h"
#include "game.h"
#include "mcts_agent_selfplay.h"
#include "logger.h"
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <chrono>
#include <iomanip>



/**
 * Self-Play Manager for Shared Memory Inference
 * Manages parallel game workers that communicate with JAX server
 */
class SelfPlayManager {
public:
    /**
     * Generate training data through self-play
     */
    void generate_training_data(
        int num_parallel_games,
        int games_per_worker,
        GameDataset& dataset,
        int mcts_simulations,
        int board_size,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        const std::string& shm_name = "mcts_jax_inference") {
        
        std::cout << "\n==============================================\n";
        std::cout << "[SelfPlay] Starting Parallel Self-Play\n";
        std::cout << "==============================================\n";
        std::cout << "[SelfPlay] num_parallel_games = " << num_parallel_games << "\n";
        std::cout << "[SelfPlay] games_per_worker   = " << games_per_worker << "\n";
        std::cout << "[SelfPlay] total_games        = " << (num_parallel_games * games_per_worker) << "\n";
        std::cout << "[SelfPlay] mcts_simulations   = " << mcts_simulations << "\n";
        std::cout << "[SelfPlay] board_size         = " << board_size << "\n";
        std::cout << "[SelfPlay] exploration_factor = " << exploration_factor << "\n";
        std::cout << "[SelfPlay] dirichlet_alpha    = " << dirichlet_alpha << "\n";
        std::cout << "[SelfPlay] dirichlet_epsilon  = " << dirichlet_epsilon << "\n";
        std::cout << "[SelfPlay] shared_memory      = " << shm_name << "\n";
        std::cout << "==============================================\n" << std::flush;
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Create shared memory queue
        auto client = std::make_shared<InferenceClient>("/mcts_jax_inference");
        
        
        // Wait for JAX server to be ready
        std::cout << "[SelfPlay] Waiting for JAX server..." << std::flush;
        int dots = 0;
        while (!client->is_server_ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (++dots % 10 == 0) std::cout << "." << std::flush;
            if (dots > 100) {
                std::cerr << "\n[SelfPlay] ERROR: Timeout waiting for JAX server (10+ seconds)" << std::endl;
                std::cerr << "[SelfPlay] Make sure inference_server.py is running!" << std::endl;
                throw std::runtime_error("JAX server not ready");
            }
        }
        std::cout << " ✓ Connected!\n" << std::flush;
        
        std::cout << "[SelfPlay] Testing connection with first inference request...\n" << std::flush;
        
        std::atomic<int> games_completed{0};
        
        // Launch game worker threads
        std::vector<std::thread> game_threads;
        game_threads.reserve(num_parallel_games);
        
        std::cout << "[SelfPlay] Starting " << num_parallel_games << " game workers...\n" << std::flush;
        
        for (int i = 0; i < num_parallel_games; ++i) {
            game_threads.emplace_back(
                &SelfPlayManager::game_worker,
                this,
                i,
                games_per_worker,
                client,  // Pass raw pointer
                &dataset,
                board_size,
                mcts_simulations,
                exploration_factor,
                dirichlet_alpha,
                dirichlet_epsilon,
                &games_completed);
            
            std::cout << "[SelfPlay] Worker " << i << " started\n" << std::flush;
        }
        
        std::cout << "[SelfPlay] All workers started, beginning self-play...\n" << std::flush;
        
        // After a few seconds, disable debug logging to reduce spam
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::cout << "[SelfPlay] Debug logging disabled (running normally)\n" << std::flush;
        
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
            size_t pending = client->pending_count();
            
            if (current > last_reported) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                float games_per_sec = (elapsed_sec > 0) ? (current / (float)elapsed_sec) : 0.0f;
                
                std::cout << "\n[SelfPlay] Progress: " << current << "/" << total_games 
                          << " games (" << std::fixed << std::setprecision(1) 
                          << (100.0 * current / total_games) << "%), "
                          << std::fixed << std::setprecision(2) << games_per_sec << " games/sec, "
                          << "queue: " << pending << " pending\n" << std::flush;
                last_reported = current;
            }
        }
        
        // Wait for all threads
        for (auto& thread : game_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        auto stats = client->get_stats();
        
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
    
    /**
     * Generate evaluation games (model1 vs model2)
     */
    struct EvaluationResults {
        int model1_wins;
        int model2_wins;
        int draws;
        int total_games;
        float model1_winrate;
        float model2_winrate;
        int eval_moves;
        int optimal_moves;
    };
    
    EvaluationResults generate_evaluation_games(
        int num_parallel_games,
        int games_per_worker,
        int mcts_simulations,
        int board_size,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        Cell_state player_to_evaluate,
        const std::string& shm_name = "mcts_jax_inference") {
        
        std::cout << "\n==============================================\n";
        std::cout << "[Evaluation] Starting Parallel Evaluation Games\n";
        std::cout << "==============================================\n";
        std::cout << "[Evaluation] num_parallel_games = " << num_parallel_games << "\n";
        std::cout << "[Evaluation] games_per_worker   = " << games_per_worker << "\n";
        std::cout << "[Evaluation] total_games        = " << (num_parallel_games * games_per_worker) << "\n";
        std::cout << "[Evaluation] mcts_simulations   = " << mcts_simulations << "\n";
        std::cout << "[Evaluation] exploration_factor = " << exploration_factor << "\n";
        std::cout << "[Evaluation] dirichlet_epsilon  = " << dirichlet_epsilon << " (should be 0.0)\n";
        std::cout << "[Evaluation] shared_memory      = " << shm_name << "\n";
        std::cout << "==============================================\n" << std::flush;
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Create shared memory queue
        auto client = std::make_shared<InferenceClient>("/mcts_jax_inference");
        
        // Wait for server
        std::cout << "[Evaluation] Waiting for JAX server..." << std::flush;
        while (!client->is_server_ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << " ✓ Connected!\n" << std::flush;
        
        std::atomic<int> games_completed{0};
        std::atomic<int> model1_wins{0};
        std::atomic<int> model2_wins{0};
        std::atomic<int> draws{0};
        std::atomic<int> eval_moves{0};
        std::atomic<int> optimal_moves{0};
        
        // Launch evaluation worker threads
        std::vector<std::thread> game_threads;
        game_threads.reserve(num_parallel_games);
        
        for (int i = 0; i < num_parallel_games; ++i) {
            game_threads.emplace_back(
                &SelfPlayManager::evaluation_game_worker,
                this,
                i,
                games_per_worker,
                client,  // Pass raw pointer
                board_size,
                mcts_simulations,
                exploration_factor,
                dirichlet_alpha,
                dirichlet_epsilon,
                player_to_evaluate,
                &optimal_moves,
                &eval_moves,
                &games_completed,
                &model1_wins,
                &model2_wins,
                &draws);
        }
        
        // Monitor progress
        int total_games = num_parallel_games * games_per_worker;
        int last_reported = 0;
        
        while (true) {
            int current = games_completed.load(std::memory_order_acquire);
            if (current >= total_games) {
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            if (current > last_reported) {
                std::cout << "[Evaluation] Progress: " << current << "/" << total_games 
                          << " games (" << (100.0 * current / total_games) << "%)\n" << std::flush;
                last_reported = current;
            }
        }
        
        // Wait for all threads
        for (auto& thread : game_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        // Calculate results
        EvaluationResults results;
        results.model1_wins = model1_wins.load();
        results.model2_wins = model2_wins.load();
        results.draws = draws.load();
        results.total_games = total_games;
        results.eval_moves = eval_moves.load();
        results.optimal_moves = optimal_moves.load();
        
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
        std::cout << "[Evaluation] Optimal moves: " << results.optimal_moves << "\n";
        std::cout << "[Evaluation] Eval moves: " << results.eval_moves << "\n";
        if (results.eval_moves > 0) {
            std::cout << "[Evaluation] Minimax Top 1 accuracy: " 
                      << (results.optimal_moves / (float)results.eval_moves) << "\n";
        }
        std::cout << "==============================================\n\n" << std::flush;
        
        return results;
    }

private:
    void game_worker(
        int worker_id,
        int num_games,
        std::shared_ptr<InferenceClient> client,
        GameDataset* dataset,
        int board_size,
        int mcts_simulations,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        std::atomic<int>* games_completed) {
        
        if (!client || !dataset || !games_completed) {
            std::cerr << "[Worker " << worker_id << "] ERROR: nullptr!" << std::endl;
            return;
        }
        
        std::cout << "[Worker " << worker_id << "] Starting with " << num_games 
                  << " games to play" << std::endl;
        
        GameDataset local_dataset(num_games * 20);
        
        for (int game_idx = 0; game_idx < num_games; ++game_idx) {
            try {
                if (game_idx == 0) {
                    std::cout << "[Worker " << worker_id << "] Creating players for game 1..." 
                              << std::endl;
                }
                
                // Both players use MODEL_1 for self-play
                auto player1 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    1.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    client,
                    -1,
                    false,
                    0);
                
                auto player2 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    1.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    client,
                    -1,
                    false,
                    0);
                
                if (game_idx == 0) {
                    std::cout << "[Worker " << worker_id << "] Players created, starting game..." 
                              << std::endl;
                }
                
                Game game(board_size, std::move(player1), std::move(player2), 
                         local_dataset, false);
                
                if (game_idx == 0) {
                    std::cout << "[Worker " << worker_id << "] Game object created, playing..." 
                              << std::endl;
                }
                
                Cell_state winner = game.play();
                
                if (game_idx == 0) {
                    std::cout << "[Worker " << worker_id << "] Game 1 completed! Winner: " 
                              << static_cast<int>(winner) << std::endl;
                }
                
                int completed = games_completed->fetch_add(1, std::memory_order_release) + 1;
                
                if (game_idx < 3 || game_idx == num_games - 1) {
                    std::cout << "[Worker " << worker_id << "] Completed game " 
                              << (game_idx + 1) << "/" << num_games 
                              << " (total: " << completed << ")" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[Worker " << worker_id << "] ERROR in game " 
                          << (game_idx + 1) << ": " << e.what() << std::endl;
            }
        }
        
        try {
            dataset->merge(local_dataset);
            std::cout << "[Worker " << worker_id << "] Finished all " << num_games 
                      << " games, data merged" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Worker " << worker_id << "] ERROR merging dataset: " 
                      << e.what() << std::endl;
        }
    }
    
    void evaluation_game_worker(
        int worker_id,
        int num_games,
        std::shared_ptr<InferenceClient> client,
        int board_size,
        int mcts_simulations,
        double exploration_factor,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        Cell_state player_to_evaluate,
        std::atomic<int>* optimal_moves,
        std::atomic<int>* eval_moves,
        std::atomic<int>* games_completed,
        std::atomic<int>* model1_wins,
        std::atomic<int>* model2_wins,
        std::atomic<int>* draws) {
        
        GameDataset dummy_dataset(20);
        
        for (int game_idx = 0; game_idx < num_games; ++game_idx) {
            try {
                // Player 1 uses MODEL_1, Player 2 uses MODEL_2
                auto player1 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    0.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    client,
                    -1,
                    false,
                    0);
                
                auto player2 = std::make_unique<Mcts_player_selfplay>(
                    exploration_factor,
                    mcts_simulations,
                    LogLevel::NONE,
                    0.0f,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    client,
                    -1,
                    false,
                    1);
                
                Game game(board_size, std::move(player1), std::move(player2), 
                         dummy_dataset, true, player_to_evaluate);
                
                Cell_state winner = game.play();
                eval_moves->fetch_add(game.get_number_eval_moves(), std::memory_order_relaxed);
                optimal_moves->fetch_add(game.get_number_optimal_moves(), std::memory_order_relaxed);
                
                if (winner == Cell_state::X) {
                    model1_wins->fetch_add(1, std::memory_order_relaxed);
                } else if (winner == Cell_state::O) {
                    model2_wins->fetch_add(1, std::memory_order_relaxed);
                } else {
                    draws->fetch_add(1, std::memory_order_relaxed);
                }
                
                games_completed->fetch_add(1, std::memory_order_release);
                
            } catch (const std::exception& e) {
                std::cerr << "[EvalWorker " << worker_id << "] ERROR: " 
                          << e.what() << std::endl;
            }
        }
    }
};