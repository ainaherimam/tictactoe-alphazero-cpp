#include "console_interface.h"

/**
 * @brief Calls the run_console_interface() function.
 */
int main() {
  run_console_interface();
  return 0;
}

// #include "selfplay_manager.h"
// #include "alphaz_model.h"
// #include "game_dataset.h"
// #include <torch/torch.h>
// #include <iostream>
// #include <memory>
// #include <iomanip>
// #include <filesystem>
// #include <ctime>

// namespace fs = std::filesystem;

// /**
//  * @brief Get current timestamp as string for file naming
//  */
// std::string get_timestamp() {
//     auto t = std::time(nullptr);
//     auto tm = *std::localtime(&t);
//     std::ostringstream oss;
//     oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
//     return oss.str();
// }

// /**
//  * @brief Main training loop with parallel self-play
//  */
// int main(int argc, char* argv[]) {
//     try {
//         std::cout << "\n" << std::string(80, '=') << std::endl;
//         std::cout << "AlphaZero Training with Parallel Self-Play" << std::endl;
//         std::cout << std::string(80, '=') << std::endl;
        
//         // ========================================
//         // CONFIGURATION PARAMETERS
//         // ========================================
        
//         // Board configuration
//         const int BOARD_SIZE = 4;
//         const int INPUT_CHANNELS = 3;  // Adjust based on your board.to_tensor()
//         const int NUM_MOVES = 16;      // Total action space size
        
//         // Network architecture
//         const int CONV_CHANNELS = 64;
//         const int NUM_RES_BLOCKS = 3;
        
//         // Self-play configuration
//         const int NUM_PARALLEL_GAMES = 12;      // Number of games to run in parallel
//         const int GAMES_PER_WORKER = 20;        // Games each worker plays per iteration
//         const int BATCH_SIZE = 12;              // GPU batch size for inference
//         const int MCTS_SIMULATIONS = 200;       // MCTS simulations per move (training)
        
//         // MCTS parameters
//         const double EXPLORATION_FACTOR = 1.0;  // PUCT exploration constant
//         const float DIRICHLET_ALPHA = 0.3f;     // Dirichlet noise alpha
//         const float DIRICHLET_EPSILON = 0.25f;  // Dirichlet noise mixing weight
        
//         // Training configuration
//         const int NUM_TRAINING_ITERATIONS = 100;    // Total training iterations
//         const int TRAINING_STEPS_PER_ITER = 1000;   // Training steps per iteration
//         const int TRAINING_BATCH_SIZE = 256;        // Batch size for network training
//         const float LEARNING_RATE = 0.001f;         // Adam optimizer learning rate
        
//         // Dataset configuration
//         const size_t DATASET_MAX_SIZE = 500000;     // Maximum dataset size (circular buffer)
        
//         // Checkpointing
//         const int CHECKPOINT_INTERVAL = 10;         // Save model every N iterations
//         const std::string CHECKPOINT_DIR = "checkpoints";
        
//         // Logging
//         const int LOG_INTERVAL = 100;               // Log training every N steps
//         const bool DETAILED_LOGGING = true;         // Show detailed loss breakdown
        
//         std::cout << "\n[Configuration]" << std::endl;
//         std::cout << "Board size: " << BOARD_SIZE << "x" << BOARD_SIZE << std::endl;
//         std::cout << "Parallel games: " << NUM_PARALLEL_GAMES << std::endl;
//         std::cout << "Games per worker: " << GAMES_PER_WORKER << std::endl;
//         std::cout << "Total games per iteration: " << (NUM_PARALLEL_GAMES * GAMES_PER_WORKER) << std::endl;
//         std::cout << "Inference batch size: " << BATCH_SIZE << std::endl;
//         std::cout << "MCTS simulations: " << MCTS_SIMULATIONS << std::endl;
//         std::cout << "Training iterations: " << NUM_TRAINING_ITERATIONS << std::endl;
//         std::cout << "Training steps per iteration: " << TRAINING_STEPS_PER_ITER << std::endl;
        
//         // ========================================
//         // DEVICE SETUP
//         // ========================================
        
//         torch::Device device = torch::kCPU;
//         if (torch::cuda::is_available()) {
//             device = torch::kCUDA;
//             std::cout << "\n[GPU Status]" << std::endl;
//             std::cout << "CUDA is available!" << std::endl;
//             std::cout << "GPU count: " << torch::cuda::device_count() << std::endl;
//             std::cout << "Using device: CUDA:0" << std::endl;
//         } else {
//             std::cout << "\n[GPU Status]" << std::endl;
//             std::cout << "WARNING: CUDA not available. Training will be VERY slow on CPU!" << std::endl;
//             std::cout << "Using device: CPU" << std::endl;
//         }
        
//         // ========================================
//         // CREATE CHECKPOINT DIRECTORY
//         // ========================================
        
//         if (!fs::exists(CHECKPOINT_DIR)) {
//             fs::create_directory(CHECKPOINT_DIR);
//             std::cout << "\n[Checkpoints]" << std::endl;
//             std::cout << "Created checkpoint directory: " << CHECKPOINT_DIR << std::endl;
//         }
        
//         // ========================================
//         // INITIALIZE NEURAL NETWORK
//         // ========================================
        
//         std::cout << "\n[Network Initialization]" << std::endl;
//         auto network = std::make_shared<AlphaZModel>(
//             INPUT_CHANNELS,
//             BOARD_SIZE,
//             BOARD_SIZE,
//             NUM_MOVES,
//             CONV_CHANNELS,
//             NUM_RES_BLOCKS
//         );
        
//         network->to(device);
//         network->eval();  // Start in evaluation mode for inference
        
//         std::cout << "Network architecture:" << std::endl;
//         std::cout << "  Input: " << INPUT_CHANNELS << " x " << BOARD_SIZE << " x " << BOARD_SIZE << std::endl;
//         std::cout << "  Convolutional channels: " << CONV_CHANNELS << std::endl;
//         std::cout << "  Residual blocks: " << NUM_RES_BLOCKS << std::endl;
//         std::cout << "  Output (policy): " << NUM_MOVES << " actions" << std::endl;
//         std::cout << "  Output (value): scalar" << std::endl;
        
//         // Count parameters
//         size_t total_params = 0;
//         for (const auto& p : network->parameters()) {
//             total_params += p.numel();
//         }
//         std::cout << "  Total parameters: " << total_params << std::endl;
        
//         // ========================================
//         // INITIALIZE TRAINING DATASET
//         // ========================================
        
//         std::cout << "\n[Dataset Initialization]" << std::endl;
//         GameDataset training_dataset(DATASET_MAX_SIZE);
//         std::cout << "Dataset maximum size: " << DATASET_MAX_SIZE << " positions" << std::endl;
        
//         // ========================================
//         // INITIALIZE SELF-PLAY MANAGER
//         // ========================================
        
//         SelfPlayManager selfplay_manager;
        
//         // ========================================
//         // MAIN TRAINING LOOP
//         // ========================================
        
//         std::cout << "\n" << std::string(80, '=') << std::endl;
//         std::cout << "Starting Training Loop" << std::endl;
//         std::cout << std::string(80, '=') << std::endl;
        
//         auto training_start_time = std::chrono::steady_clock::now();
        
//         for (int iteration = 0; iteration < NUM_TRAINING_ITERATIONS; ++iteration) {
//             std::cout << "\n" << std::string(80, '=') << std::endl;
//             std::cout << "ITERATION " << (iteration + 1) << "/" << NUM_TRAINING_ITERATIONS << std::endl;
//             std::cout << std::string(80, '=') << std::endl;
            
//             auto iter_start_time = std::chrono::steady_clock::now();
            
//             // ========================================
//             // PHASE 1: SELF-PLAY DATA GENERATION
//             // ========================================
            
//             std::cout << "\n[Phase 1: Self-Play]" << std::endl;
//             std::cout << "Generating training data through parallel self-play..." << std::endl;
            
//             size_t dataset_size_before = training_dataset.actual_size();
            
//             selfplay_manager.generate_training_data(
//                 NUM_PARALLEL_GAMES,
//                 GAMES_PER_WORKER,
//                 network,
//                 training_dataset,
//                 BATCH_SIZE,
//                 MCTS_SIMULATIONS,
//                 BOARD_SIZE,
//                 EXPLORATION_FACTOR,
//                 DIRICHLET_ALPHA,
//                 DIRICHLET_EPSILON
//             );
            
//             size_t new_positions = training_dataset.actual_size() - dataset_size_before;
//             std::cout << "\nSelf-play complete:" << std::endl;
//             std::cout << "  New positions: " << new_positions << std::endl;
//             std::cout << "  Total dataset size: " << training_dataset.actual_size() << std::endl;
            
//             // Skip training if not enough data
//             if (training_dataset.actual_size() < TRAINING_BATCH_SIZE) {
//                 std::cout << "Not enough data for training yet (need at least " 
//                           << TRAINING_BATCH_SIZE << " positions)" << std::endl;
//                 continue;
//             }
            
//             // ========================================
//             // PHASE 2: NEURAL NETWORK TRAINING
//             // ========================================
            
//             std::cout << "\n[Phase 2: Network Training]" << std::endl;
//             std::cout << "Training network on collected data..." << std::endl;
            
//             // Train the network using your existing train() function
//             train(
//                 network,
//                 training_dataset,
//                 TRAINING_BATCH_SIZE,
//                 TRAINING_STEPS_PER_ITER,
//                 LEARNING_RATE,
//                 device,
//                 LOG_INTERVAL,
//                 DETAILED_LOGGING
//             );
            
//             // Set back to evaluation mode for next self-play
//             network->eval();
            
//             // ========================================
//             // PHASE 3: CHECKPOINTING
//             // ========================================
            
//             if ((iteration + 1) % CHECKPOINT_INTERVAL == 0) {
//                 std::cout << "\n[Phase 3: Checkpointing]" << std::endl;
                
//                 std::string checkpoint_name = "model_iter_" + std::to_string(iteration + 1) + ".pt";
//                 std::string checkpoint_path = CHECKPOINT_DIR + "/" + checkpoint_name;
                
//                 network->save_model(checkpoint_path);
//                 std::cout << "Saved checkpoint: " << checkpoint_path << std::endl;
                
//                 // Also save dataset
//                 std::string dataset_path = CHECKPOINT_DIR + "/dataset_iter_" + std::to_string(iteration + 1);
//                 training_dataset.save(dataset_path);
//                 std::cout << "Saved dataset: " << dataset_path << "_*.pt" << std::endl;
//             }
            
//             // ========================================
//             // ITERATION SUMMARY
//             // ========================================
            
//             auto iter_end_time = std::chrono::steady_clock::now();
//             auto iter_duration = std::chrono::duration_cast<std::chrono::seconds>(
//                 iter_end_time - iter_start_time).count();
            
//             std::cout << "\n[Iteration Summary]" << std::endl;
//             std::cout << "Time taken: " << iter_duration << " seconds" << std::endl;
//             std::cout << "Dataset size: " << training_dataset.actual_size() << " positions" << std::endl;
            
//             // Calculate ETA
//             auto elapsed_total = std::chrono::duration_cast<std::chrono::seconds>(
//                 iter_end_time - training_start_time).count();
//             int remaining_iters = NUM_TRAINING_ITERATIONS - (iteration + 1);
//             if (iteration > 0) {
//                 float avg_time_per_iter = elapsed_total / (float)(iteration + 1);
//                 int eta_seconds = avg_time_per_iter * remaining_iters;
//                 int eta_hours = eta_seconds / 3600;
//                 int eta_mins = (eta_seconds % 3600) / 60;
//                 std::cout << "ETA: " << eta_hours << "h " << eta_mins << "m "
//                           << "(" << remaining_iters << " iterations remaining)" << std::endl;
//             }
//         }
        
//         // ========================================
//         // TRAINING COMPLETE
//         // ========================================
        
//         auto training_end_time = std::chrono::steady_clock::now();
//         auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
//             training_end_time - training_start_time).count();
        
//         std::cout << "\n" << std::string(80, '=') << std::endl;
//         std::cout << "TRAINING COMPLETE" << std::endl;
//         std::cout << std::string(80, '=') << std::endl;
        
//         std::cout << "\n[Final Statistics]" << std::endl;
//         std::cout << "Total iterations: " << NUM_TRAINING_ITERATIONS << std::endl;
//         std::cout << "Total training time: " << (total_duration / 3600) << "h " 
//                   << ((total_duration % 3600) / 60) << "m" << std::endl;
//         std::cout << "Final dataset size: " << training_dataset.actual_size() << " positions" << std::endl;
        
//         // ========================================
//         // SAVE FINAL MODEL
//         // ========================================
        
//         std::cout << "\n[Saving Final Model]" << std::endl;
//         std::string final_model_name = "model_final_" + get_timestamp() + ".pt";
//         std::string final_model_path = CHECKPOINT_DIR + "/" + final_model_name;
        
//         network->save_model(final_model_path);
//         std::cout << "Saved final model: " << final_model_path << std::endl;
        
//         // Save final dataset
//         std::string final_dataset_path = CHECKPOINT_DIR + "/dataset_final_" + get_timestamp();
//         training_dataset.save(final_dataset_path);
//         std::cout << "Saved final dataset: " << final_dataset_path << "_*.pt" << std::endl;
        
//         std::cout << "\n" << std::string(80, '=') << std::endl;
//         std::cout << "All done! Training artifacts saved to: " << CHECKPOINT_DIR << std::endl;
//         std::cout << std::string(80, '=') << std::endl;
        
//         return 0;
        
//     } catch (const std::exception& e) {
//         std::cerr << "\n[FATAL ERROR]" << std::endl;
//         std::cerr << "Exception: " << e.what() << std::endl;
//         std::cerr << "Training aborted." << std::endl;
//         return 1;
//     }
// }

/*
 * ============================================================================
 * COMPILATION INSTRUCTIONS
 * ============================================================================
 * 
 * g++ -std=c++17 -O3 \
 *     main_selfplay.cpp \
 *     selfplay_manager.cpp \
 *     inference_queue.cpp \
 *     mcts_agent_batched.cpp \
 *     game.cpp \
 *     board.cpp \
 *     alphaz_model.cpp \
 *     game_dataset.cpp \
 *     logger.cpp \
 *     -I/path/to/libtorch/include \
 *     -I/path/to/libtorch/include/torch/csrc/api/include \
 *     -L/path/to/libtorch/lib \
 *     -ltorch -ltorch_cuda -lc10 -lc10_cuda \
 *     -Wl,-rpath,/path/to/libtorch/lib \
 *     -pthread \
 *     -lstdc++fs \
 *     -o train_parallel
 * 
 * ============================================================================
 * USAGE
 * ============================================================================
 * 
 * Basic usage:
 *     ./train_parallel
 * 
 * With GPU:
 *     CUDA_VISIBLE_DEVICES=0 ./train_parallel
 * 
 * With CPU only (slow):
 *     CUDA_VISIBLE_DEVICES="" ./train_parallel
 * 
 * ============================================================================
 * EXPECTED OUTPUT
 * ============================================================================
 * 
 * The program will:
 * 1. Create a 'checkpoints/' directory
 * 2. Run 100 training iterations, each consisting of:
 *    - Self-play: 16 parallel games x 10 games each = 160 games
 *    - Training: 1000 steps on collected data
 *    - Checkpoint: Save model every 10 iterations
 * 3. Save final model and dataset
 * 
 * Example output:
 * 
 * ================================================================================
 * ITERATION 1/100
 * ================================================================================
 * 
 * [Phase 1: Self-Play]
 * Generating training data through parallel self-play...
 * 
 * === Starting Parallel Self-Play ===
 * Parallel games: 16
 * Games per worker: 10
 * Total games: 160
 * Batch size: 32
 * MCTS simulations: 200
 * ===================================
 * 
 * Launching inference thread...
 * Launching 16 game threads...
 * Progress: 32/160 games (20.0%), 2.3 games/sec, queue: 5 pending
 * Progress: 64/160 games (40.0%), 2.5 games/sec, queue: 3 pending
 * Progress: 96/160 games (60.0%), 2.4 games/sec, queue: 4 pending
 * Progress: 128/160 games (80.0%), 2.6 games/sec, queue: 2 pending
 * Progress: 160/160 games (100.0%), 2.5 games/sec, queue: 0 pending
 * 
 * === Self-Play Complete ===
 * Total games: 160
 * Total time: 64 seconds
 * Games per second: 2.5
 * Training examples collected: 4832
 * =========================
 * 
 * [Phase 2: Network Training]
 * Training network on collected data...
 * Step 100/1000 - Loss: 2.345, Policy Loss: 1.234, Value Loss: 1.111
 * Step 200/1000 - Loss: 2.123, Policy Loss: 1.100, Value Loss: 1.023
 * ...
 * 
 * ============================================================================
 * MONITORING & DEBUGGING
 * ============================================================================
 * 
 * Monitor GPU usage:
 *     watch -n 1 nvidia-smi
 * 
 * Check batch efficiency:
 *     grep "avg batch size" output.log
 *     # Should be close to BATCH_SIZE (32)
 * 
 * Check dataset growth:
 *     grep "Total dataset size" output.log
 * 
 * ============================================================================
 * TUNING RECOMMENDATIONS
 * ============================================================================
 * 
 * For faster training:
 * - Increase NUM_PARALLEL_GAMES to 24-32 (if you have CPU cores)
 * - Increase BATCH_SIZE to 64-128 (if GPU has memory)
 * - Decrease MCTS_SIMULATIONS to 100-150 (lower quality but faster)
 * 
 * For better quality:
 * - Increase MCTS_SIMULATIONS to 400-800
 * - Increase TRAINING_STEPS_PER_ITER to 2000-5000
 * - Increase NUM_RES_BLOCKS to 5-10
 * - Increase CONV_CHANNELS to 128-256
 * 
 * For memory constraints:
 * - Decrease DATASET_MAX_SIZE
 * - Decrease NUM_PARALLEL_GAMES
 * - Decrease BATCH_SIZE
 * 
 * ============================================================================
 */