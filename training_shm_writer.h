#pragma once

#include "training_shm_protocol.h"
#include "position_pool.h"
#include <string>
#include <mutex>
#include <memory>

// ============================================================================
// TRAINING SHARED MEMORY WRITER
// ============================================================================
// This class wraps the /az_training shared memory segment.
// C++ self-play workers call flush_game() to write completed games.
// Thread-safe: multiple workers can flush concurrently.
//
// Usage:
//   auto writer = std::make_shared<TrainingShmWriter>(20000);
//   // ... in game loop after commit_game() ...
//   writer->flush_game(pool, game_id);
// ============================================================================

class TrainingShmWriter {
public:
    // Constructor: creates the /az_training segment
    // @param max_capacity: ring buffer capacity (default 20,000 positions)
    // @param segment_name: SHM segment name (default "/az_training")
    explicit TrainingShmWriter(size_t max_capacity = 20000,
                              const std::string& segment_name = "/az_training");
    
    // Destructor: unlinks the shared memory segment
    ~TrainingShmWriter();
    
    // Flush a completed game from PositionPool to shared memory
    // This is called after PositionPool::commit_game() has filled in z-values
    // Thread-safe: uses internal mutex
    // @param pool: the position pool containing the game data
    // @param game_id: which game to flush
    void flush_game(const PositionPool& pool);
    
    // Get the buffer capacity
    size_t max_capacity() const { return max_capacity_; }
    
    // Signal shutdown to Python training process
    void shutdown();
    
    // Check if shutdown has been requested
    bool is_shutdown() const;
    
    // Get current statistics
    uint64_t generation() const;
    uint32_t current_size() const;
    uint32_t write_index() const;

private:
    // No copy/move
    TrainingShmWriter(const TrainingShmWriter&) = delete;
    TrainingShmWriter& operator=(const TrainingShmWriter&) = delete;
    
    std::string segment_name_;
    size_t max_capacity_;
    int shm_fd_;
    void* shm_base_;
    size_t shm_size_;
    
    TrainingBufferHeader* header_;
    TrainingPosition* positions_;
    
    // Single mutex protects all writes
    std::mutex write_mutex_;
};