#pragma once

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include "cell_state.h"

// ============================================================================
// GameLogEntry - plain data, no I/O, no allocation beyond the string
// Workers push these into the queue and move on immediately
// ============================================================================
struct GameLogEntry {
    uint64_t    game_id;        // Globally unique, atomic counter - no collisions ever
    std::string move_history;   // Already built during play()
    Cell_state  winner;
};

// ============================================================================
// GameLogger
// - Single instance, shared across all workers
// - Workers call push() which is non-blocking (just locks a mutex briefly to
//   append to queue, then continues playing the next game)
// - One dedicated writer thread drains the queue and does all file I/O
// - Naming: game_000001.pgn, game_000002.pgn ... (atomic counter, zero collisions)
// ============================================================================
class GameLogger {
public:
    // dir = output folder (e.g. "games/"), must exist before construction
    explicit GameLogger(const std::string& dir)
        : output_dir_(dir.back() == '/' ? dir : dir + "/")
        , running_(true)
        , next_game_id_(1)
    {
        writer_thread_ = std::thread(&GameLogger::writer_loop, this);
    }

    ~GameLogger() {
        running_.store(false, std::memory_order_relaxed);
        cv_.notify_one();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }

    // ======================================================================
    // Called by worker threads - non-blocking
    // ======================================================================
    void push(const std::string& move_history, Cell_state winner) {
        GameLogEntry entry;
        entry.game_id     = next_game_id_.fetch_add(1, std::memory_order_relaxed);
        entry.move_history = move_history;
        entry.winner      = winner;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(entry));
        }
        cv_.notify_one();  // Wake writer thread
    }

    // Total games pushed so far (useful for final stats)
    uint64_t total_pushed() const {
        return next_game_id_.load(std::memory_order_relaxed) - 1;
    }

    // Total games actually written to disk
    uint64_t total_written() const {
        return total_written_.load(std::memory_order_relaxed);
    }

    // Block until all queued entries are flushed to disk
    void flush_all() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty()) break;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

private:
    // ======================================================================
    // Runs on the dedicated writer thread - does ALL file I/O here
    // ======================================================================
    void writer_loop() {
        while (true) {
            GameLogEntry entry;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return !queue_.empty() || !running_.load(std::memory_order_relaxed);
                });

                // Drain remaining entries on shutdown
                if (queue_.empty() && !running_.load(std::memory_order_relaxed)) {
                    break;
                }

                entry = std::move(queue_.front());
                queue_.pop();
            }

            write_entry(entry);
        }
    }

    // ======================================================================
    // Actual file write - only called from writer thread, no locks needed
    // ======================================================================
    void write_entry(const GameLogEntry& entry) {
        std::string result =
            (entry.winner == Cell_state::X) ? "1-0" :
            (entry.winner == Cell_state::O) ? "0-1" : "1/2-1/2";

        // game_000001.pgn, game_000002.pgn, ...
        std::ostringstream filename;
        filename << output_dir_ << "game_"
                 << std::setfill('0') << std::setw(6) << entry.game_id
                 << ".pgn";

        std::ofstream outfile(filename.str());
        if (!outfile.is_open()) {
            std::cerr << "[GameLogger] ERROR: cannot open " << filename.str() << "\n";
            return;
        }

        outfile << "[Event \"AZ Selfplay\"]\n"
               << "[X \"Player 1\"]\n"
               << "[O \"Player 2\"]\n"
               << result << "\n"
               << "\n" << entry.move_history << "\n";

        outfile.close();
        total_written_.fetch_add(1, std::memory_order_relaxed);
    }

    // ======================================================================
    // Members
    // ======================================================================
    std::string                 output_dir_;
    std::thread                 writer_thread_;
    std::mutex                  mutex_;
    std::condition_variable     cv_;
    std::queue<GameLogEntry>    queue_;
    std::atomic<bool>           running_;
    std::atomic<uint64_t>       next_game_id_;      // game ID counter
    std::atomic<uint64_t>       total_written_{0};  // how many written to disk
};