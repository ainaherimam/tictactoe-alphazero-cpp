#include "core/game/game.h"
#include "core/mcts/position_pool.h"
#include "core/game/player.h"
#include <sys/stat.h>
#include "core/utils/game_logger.h"
#include "core/game/cell_state.h"
#include <iostream>
#include <memory>
#include "training/training_shm_writer.h"
#include <thread>
#include <vector>
#include "core/game/constants.h"
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;


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

    // Collect everything to move (skip existing archive_ folders).
    std::vector<fs::path> to_move;
    for (const auto& entry : fs::directory_iterator(checkpoints_dir)) {
        std::string name = entry.path().filename().string();
        if (name.rfind("archive_", 0) == 0) continue;
        to_move.push_back(entry.path());
    }

    if (to_move.empty()) {
        std::cout << "Checkpoints directory is clean - nothing to archive.\n";
        return;
    }

    // Build timestamped archive folder name: archive_YYYYMMDD_HHMMSS
    auto now        = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);

    std::ostringstream ts;
    ts << "archive_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    fs::path archive_dir = fs::path(checkpoints_dir) / ts.str();
    fs::create_directories(archive_dir);

    std::cout << "Archiving " << to_move.size() << " item(s) into " << archive_dir << "\n";
    for (const auto& src : to_move) {
        try {
            fs::rename(src, archive_dir / src.filename());
            std::cout << "  Moved: " << src.filename() << "\n";
        } catch (const fs::filesystem_error& e) {
            std::cerr << "  Warning: could not move " << src.filename()
                      << " - " << e.what() << "\n";
        }
    }
    std::cout << "Archive complete. Checkpoints directory is ready for this run.\n\n";
}


std::string get_session_dir() {
    struct stat st;
    if (stat("games", &st) != 0) {
        // games/ doesn't exist, just create it
        mkdir("games", 0755);
        return "games";
    }

    // games/ exists, create a timestamped subfolder
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_r(&time_t_now, &local_time);

    std::ostringstream dir;
    dir << "games/Selfplay-"
        << std::setfill('0')
        << std::setw(4) << (local_time.tm_year + 1900) << "-"
        << std::setw(2) << (local_time.tm_mon + 1)     << "-"
        << std::setw(2) << local_time.tm_mday          << "-"
        << std::setw(2) << local_time.tm_hour          << "-"
        << std::setw(2) << local_time.tm_min           << "-"
        << std::setw(2) << local_time.tm_sec;

    mkdir(dir.str().c_str(), 0755);
    return dir.str();
}


void selfplay_worker(int worker_id,
                     std::shared_ptr<TrainingShmWriter>  shm_writer,
                     std::shared_ptr<SharedMemoryInferenceQueue>    queue,
                     std::shared_ptr<GameLogger>         game_logger,
                     int games_per_worker) {

    if (!queue->wait_for_server(30000))
        throw std::runtime_error("Inference server not ready");

    PositionPool pool(POOL_CAPACITY);

    for (int game_num = 0; game_num < games_per_worker; ++game_num) {

        Mcts_config config(1.4, 100, LogLevel::NONE, 0.0, 0.3, 0.25, queue, -1, false, 0);
        
        auto player1 = std::make_unique<Mcts_player_selfplay>(config);
        auto player2 = std::make_unique<Mcts_player_selfplay>(config);

        Game game(std::move(player1), std::move(player2), pool, false);
        Cell_state winner = game.play();

        shm_writer->flush_game(pool);
        game_logger->push(game.get_move_history(), winner);

        pool.reset();
    }
}


int main(int argc, char** argv) {

    const size_t MAX_CAPACITY  = 50000;
    const int    NUM_WORKERS   = 4;
    const int    GAMES_PER_WORKER = 2500;

    // Archive any leftover checkpoint folders / trigger files from previous runs.
    archive_checkpoints("checkpoints/");

    std::string session_dir = get_session_dir();
    
    auto shm_writer  = std::make_shared<TrainingShmWriter>(MAX_CAPACITY);
    auto queue      = std::make_shared<SharedMemoryInferenceQueue>("/mcts_jax_inference");
    auto game_logger = std::make_shared<GameLogger>(session_dir);


    std::cout << "Waiting for best model server...\n";
    if (!queue->wait_for_server(30000)) {
        std::cerr << "Fatal: Best model inference server not ready. Exiting.\n";
        return 1;
    }

    std::vector<std::thread> workers;
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workers.emplace_back(selfplay_worker, i, shm_writer, queue, game_logger, GAMES_PER_WORKER);
    }

    for (auto& w : workers) w.join();

    // Wait for the writer thread to finish flushing everything to disk
    game_logger->flush_all();

    std::cout << "\nAll workers finished!\n";
    std::cout << "  Games pushed:  " << game_logger->total_pushed()  << "\n";
    std::cout << "  Games written: " << game_logger->total_written() << "\n";
    std::cout << "  Positions:     " << shm_writer->current_size()   << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(60));
    shm_writer->shutdown();
    return 0;
}