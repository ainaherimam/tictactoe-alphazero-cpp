#include "core/game/game.h"
#include "core/mcts/position_pool.h"
#include "core/game/player.h"
#include <sys/stat.h>
#include "core/utils/game_logger.h"
#include "core/game/cell_state.h"
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include "core/game/constants.h"
#include <chrono>
#include <atomic>
#include <mutex>
#include "inference/triton/triton_inference_client.h"

#include <filesystem>
#include <chrono>
#include <thread>
#include <iostream>

namespace fs = std::filesystem;


int wait_for_trigger(const std::string& folder_path) {
    std::cout << "Waiting for trigger..." << std::endl;
    
    while (true) {
        // Check for files matching the pattern run.evaluation.*.trigger
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // Check if filename matches pattern "run.evaluation.<number>.trigger"
                if (filename.find("run.evaluation.") == 0 && 
                    filename.find(".trigger") == filename.length() - 8) {
                    
                    // Extract the generation number
                    size_t start_pos = 15; // Length of "run.evaluation."
                    size_t end_pos = filename.length() - 8; // Remove ".trigger"
                    std::string gen_str = filename.substr(start_pos, end_pos - start_pos);
                    
                    try {
                        int current_gen = std::stoi(gen_str);
                        
                        // Delete the trigger file
                        try {
                            fs::remove(entry.path());
                            std::cout << "Trigger detected * Launching evaluation for generation " 
                                      << current_gen << std::endl;
                        } catch (const fs::filesystem_error& e) {
                            std::cerr << "Failed to delete trigger: " << e.what() << std::endl;
                        }
                        
                        return current_gen;
                        
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid generation number in trigger file: " 
                                  << filename << std::endl;
                    }
                }
            }
        }
        
        // Poll every 500ms
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


struct EvaluationResults {
    std::atomic<int> best_as_p1_wins{0};
    std::atomic<int> best_as_p1_losses{0};
    std::atomic<int> best_as_p1_draws{0};
    
    std::atomic<int> best_as_p2_wins{0};
    std::atomic<int> best_as_p2_losses{0};
    std::atomic<int> best_as_p2_draws{0};
    
    std::mutex cout_mutex;
    
    void print_stats() {
        std::lock_guard<std::mutex> lock(cout_mutex);
        
        int total_games = best_as_p1_wins + best_as_p1_losses + best_as_p1_draws +
                         best_as_p2_wins + best_as_p2_losses + best_as_p2_draws;
        
        int best_total_wins = best_as_p1_wins + best_as_p2_wins;
        int candidate_total_wins = best_as_p1_losses + best_as_p2_losses;
        int total_draws = best_as_p1_draws + best_as_p2_draws;
        
        double candidate_winrate = 0.0;
        if (total_games > 0) {
            candidate_winrate = (static_cast<double>(candidate_total_wins) / 
                                (total_games - total_draws)) * 100.0;
        }
        
        std::cout << "\n=== Evaluation Results ===\n";
        std::cout << "Total games completed: " << total_games << " / 400\n";
        std::cout << "\nBest model as Player 1:\n";
        std::cout << "  Wins:   " << best_as_p1_wins << "\n";
        std::cout << "  Losses: " << best_as_p1_losses << "\n";
        std::cout << "  Draws:  " << best_as_p1_draws << "\n";
        
        std::cout << "\nBest model as Player 2:\n";
        std::cout << "  Wins:   " << best_as_p2_wins << "\n";
        std::cout << "  Losses: " << best_as_p2_losses << "\n";
        std::cout << "  Draws:  " << best_as_p2_draws << "\n";
        
        std::cout << "\nOverall:\n";
        std::cout << "  Best model wins:      " << best_total_wins << "\n";
        std::cout << "  Candidate model wins: " << candidate_total_wins << "\n";
        std::cout << "  Draws:                " << total_draws << "\n";
        std::cout << "  Candidate winrate:    " << std::fixed << std::setprecision(2) 
                  << candidate_winrate << "%\n";
        std::cout << "========================\n\n";
    }
};

void eval_worker_best_as_p1(int worker_id,
                            InferenceClient* client_best,
                            InferenceClient* client_candidate,
                            std::shared_ptr<EvaluationResults> results,
                            int games_per_worker) {
    
    PositionPool dummy_pool(100);  // Dummy pool, won't be used for training
    
    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        // Best model as Player 1, Candidate model as Player 2
        Mcts_triton_config config_best(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, client_best, -1, false, 0);
        Mcts_triton_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, client_candidate, -1, false, 0);
        
        auto player1 = std::make_unique<Mcts_player_triton>(config_best);
        auto player2 = std::make_unique<Mcts_player_triton>(config_candidate);
        
        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();
        
        if (winner == Cell_state::X) {
            results->best_as_p1_wins++;
        } else if (winner == Cell_state::O) {
            results->best_as_p1_losses++;
        } else {
            results->best_as_p1_draws++;
        }
        
        dummy_pool.reset();
        
        // Print progress every 10 games
        if ((game_num + 1) % 10 == 0) {
            std::lock_guard<std::mutex> lock(results->cout_mutex);
            std::cout << "Worker " << worker_id << " (Best as P1): " 
                      << (game_num + 1) << "/" << games_per_worker << " games completed\n";
        }
    }
}

void eval_worker_best_as_p2(int worker_id,
                            InferenceClient* client_best,
                            InferenceClient* client_candidate,
                            std::shared_ptr<EvaluationResults> results,
                            int games_per_worker) {
    
    PositionPool dummy_pool(100);  // Dummy pool, won't be used for training
    
    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        // Candidate model as Player 1, Best model as Player 2
        Mcts_triton_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, client_candidate, -1, false, 0);
        Mcts_triton_config config_best(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, client_best, -1, false, 0);
        
        auto player1 = std::make_unique<Mcts_player_triton>(config_candidate);
        auto player2 = std::make_unique<Mcts_player_triton>(config_best);
        
        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();
        
        if (winner == Cell_state::O) {
            results->best_as_p2_wins++;
        } else if (winner == Cell_state::X) {
            results->best_as_p2_losses++;
        } else {
            results->best_as_p2_draws++;
        }
        
        dummy_pool.reset();
        
        // Print progress every 10 games
        if ((game_num + 1) % 10 == 0) {
            std::lock_guard<std::mutex> lock(results->cout_mutex);
            std::cout << "Worker " << worker_id << " (Best as P2): " 
                      << (game_num + 1) << "/" << games_per_worker << " games completed\n";
        }
    }
}

int main(int argc, char** argv) {
    const int NUM_WORKERS = 2;
    const int GAMES_PER_WORKER_PER_SIDE = 25;  // 25 games * 8 workers = 200 games per side
    
    std::cout << "=== Model Evaluation Loop ===\n";
    std::cout << "Starting continuous evaluation loop...\n";
    std::cout << "Press Ctrl+C to stop\n\n";
    
    while (true) {
        std::cout << "\n========================================\n";
        std::cout << "Waiting for new checkpoint trigger...\n";
        std::cout << "========================================\n";
        
        int checkpoint_number = wait_for_trigger("checkpoints/");
        
        std::cout << "\n=== Starting Evaluation for Checkpoint " << checkpoint_number << " ===\n";
        std::cout << "Initializing Triton inference clients...\n";
        
        // Initialize Triton inference clients
        std::string server_url = "localhost:8001";  // Triton gRPC endpoint
        std::string best_model_name = "mcts_best_model";
        std::string candidate_model_name = "mcts_candidate_model";
        
        InferenceClient client_best(server_url, best_model_name, 1000, 3);
        InferenceClient client_candidate(server_url, candidate_model_name, 1000, 3);
        
        std::cout << "Best model client ready!\n";
        std::cout << "Candidate model client ready!\n";
        
        auto results = std::make_shared<EvaluationResults>();
        
        std::cout << "\nStarting evaluation: 200 games with best as P1, 200 games with best as P2\n";
        std::cout << "Using " << NUM_WORKERS << " workers per side\n\n";
        
        std::vector<std::thread> workers;
        
        // Spawn workers for best model as Player 1
        for (int i = 0; i < NUM_WORKERS; ++i) {
            workers.emplace_back(eval_worker_best_as_p1, i, &client_best, &client_candidate, 
                               results, GAMES_PER_WORKER_PER_SIDE);
        }
        
        // Spawn workers for best model as Player 2
        for (int i = 0; i < NUM_WORKERS; ++i) {
            workers.emplace_back(eval_worker_best_as_p2, i + NUM_WORKERS, &client_best, 
                               &client_candidate, results, GAMES_PER_WORKER_PER_SIDE);
        }
        
        // Wait for all workers to finish
        for (auto& w : workers) {
            w.join();
        }
        
        // Print final results
        results->print_stats();
        
        // Calculate final candidate winrate (excluding draws)
        int total_decisive_games = results->best_as_p1_wins + results->best_as_p1_losses +
                                   results->best_as_p2_wins + results->best_as_p2_losses;
        int candidate_wins = results->best_as_p1_losses + results->best_as_p2_losses;
        
        double candidate_winrate = 0.0;
        if (total_decisive_games > 0) {
            candidate_winrate = (static_cast<double>(candidate_wins) / total_decisive_games) * 100.0;
        }
        
        std::cout << "\n=== FINAL VERDICT (Checkpoint " << checkpoint_number << ") ===\n";
        std::cout << "Candidate model winrate: " << std::fixed << std::setprecision(2) 
                  << candidate_winrate << "%\n";
        
        if (candidate_winrate > 55.0) {
            std::cout << "✓ CANDIDATE MODEL ACCEPTED (winrate > 55%)\n";
            std::cout << "  The candidate model outperforms the best model!\n";
            
            // Log to file
            std::ofstream log_file("evaluation_results.txt", std::ios::app);
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            log_file << "===================================\n";
            log_file << "Timestamp: " << std::ctime(&time_t_now);
            log_file << "Checkpoint: " << checkpoint_number << "\n";
            log_file << "Candidate winrate: " << candidate_winrate << "%\n";
            log_file << "Result: ACCEPTED (> 55%)\n";
            log_file << "Total games: " << total_decisive_games << " (decisive)\n";
            log_file << "Candidate wins: " << candidate_wins << "\n";
            log_file << "Best wins: " << (results->best_as_p1_wins + results->best_as_p2_wins) << "\n";
            log_file << "===================================\n\n";
            log_file.close();

            std::string trigger_filename = "checkpoints/reload.mcts_best_model." + std::to_string(checkpoint_number) + ".trigger";
            std::ofstream trigger_file(trigger_filename);
            if (trigger_file.is_open()) {
                trigger_file << "Trigger for generation " << checkpoint_number << "\n";
                trigger_file.close();
                std::cout << "  Created trigger file: " << trigger_filename << "\n";
            } else {
                std::cerr << "  Failed to create trigger file: " << trigger_filename << "\n";
            }
            
        } else {
            std::cout << "✗ CANDIDATE MODEL REJECTED (winrate ≤ 55%)\n";
            std::cout << "  The best model remains superior.\n";
            
            // Log rejection to file
            std::ofstream log_file("evaluation_results.txt", std::ios::app);
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            log_file << "===================================\n";
            log_file << "Timestamp: " << std::ctime(&time_t_now);
            log_file << "Checkpoint: " << checkpoint_number << "\n";
            log_file << "Candidate winrate: " << candidate_winrate << "%\n";
            log_file << "Result: REJECTED (<= 55%)\n";
            log_file << "Total games: " << total_decisive_games << " (decisive)\n";
            log_file << "Candidate wins: " << candidate_wins << "\n";
            log_file << "Best wins: " << (results->best_as_p1_wins + results->best_as_p2_wins) << "\n";
            log_file << "===================================\n\n";
            log_file.close();
        }
        std::cout << "=====================\n";
        
        std::cout << "\nEvaluation complete. Returning to wait for next trigger...\n";
    }
    
    return 0;
}