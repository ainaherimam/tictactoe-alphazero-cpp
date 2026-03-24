#pragma once

// TrainingPosition, TrainingBufferHeader, training_segment_size(),
// and training_positions_ptr() are all defined in the protocol header.
// This is the single source of truth — do NOT redefine them here.
#include "training/training_shm_protocol.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

// Forward declarations — full definitions pulled in by training_shm_writer.cpp
class PositionPool;   // core/mcts/position_pool.h
struct Position;      // core/game/position.h

// ============================================================================
//  TrainingShmWriter
// ============================================================================

/// Creates and owns a POSIX shared-memory segment that the Python training
/// script consumes via TrainingShmReader.
///
/// On construction the writer tries to reload any positions that were
/// previously saved to "data/last_data.bin"; if the file does not exist the
/// buffer starts empty.  Call dump_to_disk() to atomically overwrite that
/// same file so the next run can resume immediately.
class TrainingShmWriter {
public:
    /// @param max_capacity   Ring-buffer capacity (number of TrainingPositions)
    /// @param segment_name   POSIX SHM name, e.g. "/az_training"
    explicit TrainingShmWriter(size_t             max_capacity,
                               const std::string& segment_name = "/az_training");

    ~TrainingShmWriter();

    // Non-copyable, non-movable (owns a raw SHM file descriptor + mmap).
    TrainingShmWriter(const TrainingShmWriter&)            = delete;
    TrainingShmWriter& operator=(const TrainingShmWriter&) = delete;
    TrainingShmWriter(TrainingShmWriter&&)                 = delete;
    TrainingShmWriter& operator=(TrainingShmWriter&&)      = delete;

    // ── Write path ──────────────────────────────────────────────────────────

    /// Copy all positions in @p pool into the ring buffer and bump the
    /// generation counter so the Python reader knows fresh data arrived.
    /// Thread-safe: acquires write_mutex_ internally.
    void flush_game(const PositionPool& pool);

    // ── Persistence ─────────────────────────────────────────────────────────

    /// Snapshot the current ring-buffer contents to "data/last_data.bin".
    /// Thread-safe: acquires write_mutex_ internally.
    void dump_to_disk();

    // ── Queries ─────────────────────────────────────────────────────────────

    /// Ring-buffer capacity (number of TrainingPositions).
    size_t   max_capacity()  const { return max_capacity_; }

    // ── Control signals ─────────────────────────────────────────────────────

    void     shutdown();
    bool     is_shutdown()   const;
    uint64_t generation()    const;
    uint32_t current_size()  const;
    uint32_t write_index()   const;

private:
    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Called once from the constructor: reads "data/last_data.bin" and, if
    /// the file exists and is valid, pre-populates the ring buffer with its
    /// contents.  Errors are non-fatal; the writer simply starts empty.
    void load_from_disk();

    /// Writes the ring-buffer snapshot to "data/last_data.bin".
    /// Caller MUST hold write_mutex_.
    void dump_to_disk_locked();

    // ── Data members ────────────────────────────────────────────────────────

    std::string           segment_name_;
    size_t                max_capacity_;
    size_t                shm_size_;

    int                   shm_fd_;
    void*                 shm_base_;
    TrainingBufferHeader* header_;
    TrainingPosition*     positions_;

    std::mutex            write_mutex_;
};