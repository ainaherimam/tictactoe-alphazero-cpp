#include "core/game/game.h"
#include "core/mcts/position_pool.h"
#include "core/game/player.h"
#include <sys/stat.h>
#include "core/utils/game_logger.h"
#include "core/game/cell_state.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include "core/game/constants.h"
#include <chrono>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <sstream>

#include <filesystem>

namespace fs = std::filesystem;


// ---------------------------------------------------------------------------
// Scan the folder for all run.evaluation.<gen>.trigger files.
// Blocks (polls every 500ms) until at least one is found.
// Picks the lowest generation number, deletes that file, and returns the gen.
// Any other trigger files are left untouched for the next call.
// ---------------------------------------------------------------------------
int wait_for_trigger(const std::string& folder_path) {
    std::cout << "Waiting for trigger..." << std::endl;

    while (true) {
        int    lowest_gen  = -1;
        fs::path lowest_path;

        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();

            if (filename.find("run.evaluation.") != 0) continue;
            if (filename.size() <= 8)                  continue;
            if (filename.compare(filename.size() - 8, 8, ".trigger") != 0) continue;

            size_t start_pos = 15;                   // len("run.evaluation.")
            size_t end_pos   = filename.size() - 8;  // strip ".trigger"
            std::string gen_str = filename.substr(start_pos, end_pos - start_pos);

            try {
                int gen = std::stoi(gen_str);
                if (lowest_gen == -1 || gen < lowest_gen) {
                    lowest_gen  = gen;
                    lowest_path = entry.path();
                }
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid generation number in trigger file: "
                          << filename << std::endl;
            }
        }

        if (lowest_gen != -1) {
            // Delete only the chosen (lowest) trigger file.
            try {
                fs::remove(lowest_path);
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Failed to delete trigger: " << e.what() << std::endl;
                // Don't return — retry next poll so we don't process a file we
                // couldn't remove (would be picked again next iteration anyway).
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            std::cout << "Trigger detected * Launching evaluation for generation "
                      << lowest_gen << std::endl;
            return lowest_gen;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


// ---------------------------------------------------------------------------
// EvaluationResults
// ---------------------------------------------------------------------------
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

        int best_total_wins      = best_as_p1_wins   + best_as_p2_wins;
        int candidate_total_wins = best_as_p1_losses + best_as_p2_losses;
        int total_draws          = best_as_p1_draws  + best_as_p2_draws;

        double candidate_winrate = 0.0;
        if (total_games - total_draws > 0) {
            candidate_winrate = (static_cast<double>(candidate_total_wins) /
                                 (total_games - total_draws)) * 100.0;
        }

        std::cout << "\n=== Evaluation Results ===\n";
        std::cout << "Total games completed: " << total_games << " / 400\n";
        std::cout << "\nBest model as Player 1:\n";
        std::cout << "  Wins:   " << best_as_p1_wins   << "\n";
        std::cout << "  Losses: " << best_as_p1_losses << "\n";
        std::cout << "  Draws:  " << best_as_p1_draws  << "\n";

        std::cout << "\nBest model as Player 2:\n";
        std::cout << "  Wins:   " << best_as_p2_wins   << "\n";
        std::cout << "  Losses: " << best_as_p2_losses << "\n";
        std::cout << "  Draws:  " << best_as_p2_draws  << "\n";

        std::cout << "\nOverall:\n";
        std::cout << "  Best model wins:      " << best_total_wins      << "\n";
        std::cout << "  Candidate model wins: " << candidate_total_wins << "\n";
        std::cout << "  Draws:                " << total_draws          << "\n";
        std::cout << "  Candidate winrate:    " << std::fixed << std::setprecision(2)
                  << candidate_winrate << "%\n";
        std::cout << "========================\n\n";
    }
};


// ---------------------------------------------------------------------------
// Worker threads
// ---------------------------------------------------------------------------
void eval_worker_best_as_p1(int worker_id,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                             std::shared_ptr<EvaluationResults> results,
                             int games_per_worker)
{
    PositionPool dummy_pool(100);

    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        Mcts_config config_best     (1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_best,      -1, false, 0);
        Mcts_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_candidate, -1, false, 0);

        auto player1 = std::make_unique<Mcts_player_selfplay>(config_best);
        auto player2 = std::make_unique<Mcts_player_selfplay>(config_candidate);

        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();

        if      (winner == Cell_state::X) results->best_as_p1_wins++;
        else if (winner == Cell_state::O) results->best_as_p1_losses++;
        else                              results->best_as_p1_draws++;

        dummy_pool.reset();

        if ((game_num + 1) % 10 == 0) {
            std::lock_guard<std::mutex> lock(results->cout_mutex);
            std::cout << "Worker " << worker_id << " (Best as P1): "
                      << (game_num + 1) << "/" << games_per_worker << " games completed\n";
        }
    }
}

void eval_worker_best_as_p2(int worker_id,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_best,
                             std::shared_ptr<SharedMemoryInferenceQueue> queue_candidate,
                             std::shared_ptr<EvaluationResults> results,
                             int games_per_worker)
{
    PositionPool dummy_pool(100);

    for (int game_num = 0; game_num < games_per_worker; ++game_num) {
        Mcts_config config_candidate(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_candidate, -1, false, 0);
        Mcts_config config_best     (1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.0, queue_best,      -1, false, 0);

        auto player1 = std::make_unique<Mcts_player_selfplay>(config_candidate);
        auto player2 = std::make_unique<Mcts_player_selfplay>(config_best);

        Game game(std::move(player1), std::move(player2), dummy_pool, false);
        Cell_state winner = game.play();

        if      (winner == Cell_state::O) results->best_as_p2_wins++;
        else if (winner == Cell_state::X) results->best_as_p2_losses++;
        else                              results->best_as_p2_draws++;

        dummy_pool.reset();

        if ((game_num + 1) % 10 == 0) {
            std::lock_guard<std::mutex> lock(results->cout_mutex);
            std::cout << "Worker " << worker_id << " (Best as P2): "
                      << (game_num + 1) << "/" << games_per_worker << " games completed\n";
        }
    }
}


// ---------------------------------------------------------------------------
// Helper: write a sentinel / trigger file.
// ---------------------------------------------------------------------------
static void write_trigger(const std::string& path, int gen) {
    std::ofstream f(path);
    if (f.is_open()) {
        f << "Trigger for generation " << gen << "\n";
        f.close();
        std::cout << "  Created trigger file: " << path << "\n";
    } else {
        std::cerr << "  Failed to create trigger file: " << path << "\n";
    }
}


// ---------------------------------------------------------------------------
// Archive all existing folders and trigger files in the checkpoints directory
// into a timestamped sub-folder, so the directory is clean for this run.
// ---------------------------------------------------------------------------
static void archive_checkpoints(const std::string& checkpoints_dir) {
    if (!fs::exists(checkpoints_dir)) {
        fs::create_directories(checkpoints_dir);
        std::cout << "Created checkpoints directory: " << checkpoints_dir << "\n";
        return;
    }

    // Collect everything we want to move (folders and files) before iterating.
    std::vector<fs::path> to_move;
    for (const auto& entry : fs::directory_iterator(checkpoints_dir)) {
        // Skip anything that is itself an archive folder (starts with "archive_")
        std::string name = entry.path().filename().string();
        if (name.rfind("archive_", 0) == 0) continue;
        to_move.push_back(entry.path());
    }

    if (to_move.empty()) {
        std::cout << "Checkpoints directory is already clean — nothing to archive.\n";
        return;
    }

    // Build a timestamp string: archive_YYYYMMDD_HHMMSS
    auto now        = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif
    std::ostringstream ts;
    ts << "archive_"
       << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    fs::path archive_dir = fs::path(checkpoints_dir) / ts.str();

    fs::create_directories(archive_dir);
    std::cout << "Archiving " << to_move.size() << " item(s) → " << archive_dir << "\n";

    for (const auto& src : to_move) {
        fs::path dst = archive_dir / src.filename();
        try {
            fs::rename(src, dst);
            std::cout << "  Moved: " << src.filename() << "\n";
        } catch (const fs::filesystem_error& e) {
            std::cerr << "  Warning: could not move " << src.filename()
                      << " — " << e.what() << "\n";
        }
    }

    std::cout << "Archive complete. Checkpoints directory is ready for this run.\n\n";
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const std::string CHECKPOINTS_DIR       = "checkpoints/";
    const int         NUM_WORKERS           = 4;
    const int         GAMES_PER_WORKER_SIDE = 50;  // 4 workers × 50 = 200 games per side

    std::cout << "=== Model Evaluation Loop ===\n";

    // Archive any leftover folders / trigger files from previous runs.
    archive_checkpoints(CHECKPOINTS_DIR);

    std::cout << "Starting continuous evaluation loop...\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    auto queue_best      = std::make_shared<SharedMemoryInferenceQueue>("/mcts_best_model");
    auto queue_candidate = std::make_shared<SharedMemoryInferenceQueue>("/mcts_candidate_model");

    std::cout << "Waiting for best model server...\n";
    if (!queue_best->wait_for_server(30000)) {
        std::cerr << "Fatal: Best model inference server not ready. Exiting.\n";
        return 1;
    }
    std::cout << "Best model server ready!\n";

    std::cout << "Waiting for candidate model server...\n";
    if (!queue_candidate->wait_for_server(30000)) {
        std::cerr << "Fatal: Candidate model inference server not ready. Exiting.\n";
        return 1;
    }
    std::cout << "Candidate model server ready!\n\n";

    while (true) {
        // ------------------------------------------------------------------
        // 1. Block until a trigger appears; pick the lowest generation.
        // ------------------------------------------------------------------
        int gen = wait_for_trigger(CHECKPOINTS_DIR);

        std::cout << "\n=== Starting Evaluation for Checkpoint " << gen << " ===\n";

        // ------------------------------------------------------------------
        // 2. Tell the candidate server to load this checkpoint, then wait
        //    until it signals readiness.
        // ------------------------------------------------------------------
        write_trigger(CHECKPOINTS_DIR +
            "reload.mcts_candidate_model." + std::to_string(gen) + ".trigger", gen);

        std::cout << "Waiting for candidate model to reload checkpoint " << gen << "...\n";
        if (!queue_candidate->wait_for_server(60000)) {
            std::cerr << "Error: Candidate model did not become ready after reload trigger.\n";
            std::cerr << "Skipping checkpoint " << gen << " and continuing...\n";
            continue;   // go back to wait_for_trigger
        }
        std::cout << "Candidate model ready with checkpoint " << gen << "!\n";

        // ------------------------------------------------------------------
        // 3. Run evaluation.
        // ------------------------------------------------------------------
        auto results = std::make_shared<EvaluationResults>();

        std::cout << "\nStarting evaluation: 200 games with best as P1, "
                     "200 games with best as P2\n";
        std::cout << "Using " << NUM_WORKERS << " workers per side\n\n";

        std::vector<std::thread> workers;

        for (int i = 0; i < NUM_WORKERS; ++i)
            workers.emplace_back(eval_worker_best_as_p1,
                                 i, queue_best, queue_candidate,
                                 results, GAMES_PER_WORKER_SIDE);

        for (int i = 0; i < NUM_WORKERS; ++i)
            workers.emplace_back(eval_worker_best_as_p2,
                                 i + NUM_WORKERS, queue_best, queue_candidate,
                                 results, GAMES_PER_WORKER_SIDE);

        for (auto& w : workers) w.join();

        results->print_stats();

        // ------------------------------------------------------------------
        // 4. Compute final winrate.
        // ------------------------------------------------------------------
        int total_decisive = results->best_as_p1_wins  + results->best_as_p1_losses +
                             results->best_as_p2_wins  + results->best_as_p2_losses;
        int candidate_wins = results->best_as_p1_losses + results->best_as_p2_losses;

        double candidate_winrate = 0.0;
        if (total_decisive > 0)
            candidate_winrate = (static_cast<double>(candidate_wins) / total_decisive) * 100.0;

        // ------------------------------------------------------------------
        // 5. Log to file.
        // ------------------------------------------------------------------
        {
            std::ofstream log_file("evaluation_results.txt", std::ios::app);
            auto now        = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            log_file << "===================================\n";
            log_file << "Timestamp:         " << std::ctime(&time_t_now);
            log_file << "Checkpoint:        " << gen << "\n";
            log_file << "Candidate winrate: " << candidate_winrate << "%\n";
            log_file << "Total games:       " << total_decisive << " (decisive)\n";
            log_file << "Candidate wins:    " << candidate_wins << "\n";
            log_file << "Best wins:         " << (results->best_as_p1_wins + results->best_as_p2_wins) << "\n";
            log_file << "Result:            "
                     << (candidate_winrate > 55.0 ? "ACCEPTED (> 55%)" : "REJECTED (<= 55%)") << "\n";
            log_file << "===================================\n\n";
        }

        // ------------------------------------------------------------------
        // 6. Verdict.
        // ------------------------------------------------------------------
        std::cout << "\n=== FINAL VERDICT (Checkpoint " << gen << ") ===\n";
        std::cout << "Candidate model winrate: " << std::fixed << std::setprecision(2)
                  << candidate_winrate << "%\n";

        if (candidate_winrate > 55.0) {
            std::cout << "✔ CANDIDATE MODEL ACCEPTED (winrate > 55%)\n";
            std::cout << "  The candidate model outperforms the best model!\n";
            write_trigger(CHECKPOINTS_DIR +
                "reload.mcts_best_model." + std::to_string(gen) + ".trigger", gen);
        } else {
            std::cout << "✗ CANDIDATE MODEL REJECTED (winrate ≤ 55%)\n";
            std::cout << "  The best model remains superior.\n";
        }

        std::cout << "=====================\n";
        std::cout << "Evaluation complete. Checking for more triggers...\n";
        // Loop back to wait_for_trigger — picks up any accumulated triggers.
    }

    return 0;
}
