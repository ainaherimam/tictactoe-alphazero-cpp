#include "game.h"
#include "position_pool.h"
#include "player.h"
#include "board.h"
#include "cell_state.h"
#include <iostream>
#include <memory>
#include "training_shm_writer.h"
#include <thread>
#include <vector>
#include "constants.h"
#include <chrono>

void selfplay_worker(int worker_id, 
                     std::shared_ptr<TrainingShmWriter> shm_writer,
                     std::shared_ptr<InferenceClient> client,
                     int games_per_worker) {
    
    std::cout << "[Worker " << worker_id << "] Starting..." << std::endl;

    if (!client->wait_for_server(30000)) {
        throw std::runtime_error("Inference server not ready");
    }
    
    // Each worker has its own local position pool (thread-local)
    // This pool buffers positions during a game
    
    PositionPool pool(POOL_CAPACITY);
    
    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        // Create players (your existing code)
        auto player1 = std::make_unique<Mcts_player_selfplay>(1.4, 
            100, LogLevel::NONE, 0.0, 
            0.3, 0.25, client, -1, 
            false, 0);
        auto player2 = std::make_unique<Mcts_player_selfplay>(1.4, 
            100, LogLevel::NONE, 0.0, 
            0.3, 0.25, client, -1, 
            false, 0);
        

        Game game(std::move(player1), std::move(player2), pool, false);
        Cell_state winner = game.play();
        
        shm_writer->flush_game(pool);
        pool.reset();

    }
    
    std::cout << "[Worker " << worker_id << "] Finished" << std::endl;
}

void test_position_pool() {
    std::cout << "=== Testing Position Pool ===" << std::endl;
    auto shm_writer = std::make_shared<TrainingShmWriter>(20000);
    // Create a position pool with capacity for 1000 positions
    const size_t POOL_CAPACITY = 50;
    PositionPool pool(POOL_CAPACITY);
    
    std::cout << "Created PositionPool with capacity: " << POOL_CAPACITY << std::endl;

    std::shared_ptr<InferenceClient> client = std::make_shared<InferenceClient>("/mcts_jax_inference");
    if (!client->wait_for_server(30000)) {  // 30 second timeout
        throw std::runtime_error("Inference server not ready");
    }
    
    // Play multiple games
    const int NUM_GAMES = 1;
    int x_wins = 0, o_wins = 0, draws = 0;
    
    
    for (int game_num = 0; game_num < NUM_GAMES; ++game_num) {
        std::cout << "\n--- Game " << (game_num + 1) << " ---" << std::endl;
        
        auto player1 = std::make_unique<Mcts_player_selfplay>(1.4, 
            100, LogLevel::NONE, 0.0, 
            0.3, 0.25, client, -1, 
            false, 0);
        auto player2 = std::make_unique<Mcts_player_selfplay>(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.25, client, -1, false, 0);
        
        Game game(std::move(player1), std::move(player2), pool, false);
        
        Cell_state winner = game.play();

        game.play();
        
        std::string result;
        if (winner == Cell_state::X) {
            result = "X wins";
            x_wins++;
        } else if (winner == Cell_state::O) {
            result = "O wins";
            o_wins++;
        } else {
            result = "Draw";
            draws++;
        }
        
        std::cout << "Result: " << result << std::endl;
        std::cout << "Pool size: " << pool.size() << "/" << std::endl;
        
    }
    shm_writer->flush_game(pool);
    shm_writer->shutdown();
    
    
    /* // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Games played: " << NUM_GAMES << std::endl;
    std::cout << "X wins: " << x_wins << std::endl;
    std::cout << "O wins: " << o_wins << std::endl;
    std::cout << "Draws: " << draws << std::endl;
    std::cout << "Total positions collected: " << pool.size() << std::endl;
    
    // Verify positions have correct z-values
    std::cout << "\n=== Verifying Position Data ===" << std::endl;
    
    for (uint32_t game_id = 0; game_id < pool.size(); ++game_id) {
        auto num_moves = pool.size();
        
        std::cout << "Game " << game_id << ": " 
                  << num_moves << " moves starting at index " << game_id << std::endl;
        
        // Check a few positions from this game
        if (num_moves > 0) {
            const Position& first_pos = pool.get_position(game_id);
            std::cout << "  First move - player_index: " << (int)first_pos.player_index 
                      << ", z-value: " << first_pos.z << std::endl;
            
            if (num_moves > 1) {
                const Position& second_pos = pool.get_position(game_id + 1);
                std::cout << "  Second move - player_index: " << (int)second_pos.player_index 
                          << ", z-value: " << second_pos.z << std::endl;
            }
        }
    }
    
    // Test pool reset
    std::cout << "\n=== Testing Pool Reset ===" << std::endl;
    std::cout << "Before reset - Pool size: " << pool.size() << std::endl;
    pool.reset();
    std::cout << "After reset - Pool size: " << pool.size() << std::endl;
    
    std::cout << "\n=== All Tests Completed Successfully ===" << std::endl; */
}


int main(int argc, char** argv) {

    // test_position_pool();
    // Configuration
     const size_t MAX_CAPACITY = 20000;     // 20K positions in SHM (7.3 MB)
    const int NUM_WORKERS = 8;             // One per core on i5-13000
    const int GAMES_PER_WORKER = 20;     // Total: 20K games
    
    std::cout << "Starting AlphaZero Self-Play Training\n";
    std::cout << "======================================\n";
    std::cout << "Workers: " << NUM_WORKERS << "\n";
    std::cout << "Games per worker: " << GAMES_PER_WORKER << "\n";
    std::cout << "SHM capacity: " << MAX_CAPACITY << " positions\n\n";
    
    // Create shared memory writer (shared by all workers)
    auto shm_writer = std::make_shared<TrainingShmWriter>(MAX_CAPACITY);

    // Shared inference
    std::shared_ptr<InferenceClient> client = std::make_shared<InferenceClient>("/mcts_jax_inference");

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workers.emplace_back(selfplay_worker, i, shm_writer, client, GAMES_PER_WORKER);
    }
    
    // Optional: Monitor progress in main thread
    std::cout << "\nSelf-play in progress...\n";
    std::cout << "You can now run: python train.py\n\n";
    
    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }
    
    std::cout << "\nAll workers finished!\n";
    std::cout << "Final stats:\n";
    std::cout << "  Generation: " << shm_writer->generation() << "\n";
    std::cout << "  Positions: " << shm_writer->current_size() << "\n";
    
    // Signal shutdown to Python
    std::this_thread::sleep_for(std::chrono::seconds(60));
    shm_writer->shutdown();
    
    return 0;
}