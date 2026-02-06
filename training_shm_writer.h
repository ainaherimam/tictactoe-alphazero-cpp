#pragma once

#include "training_shm_protocol.h"
#include "position_pool.h"
#include <string>
#include <mutex>
#include <memory>

/**
 * @class TrainingShmWriter
 * @brief Manages writing training data to shared memory for Python consumer.
 * 
 * This class wraps the /az_training shared memory segment. C++ self-play workers
 * call flush_game() to write completed games to a ring buffer that can be consumed
 * by a Python training process.
 * 
 * Thread-safe: Multiple workers can flush games concurrently using internal mutex.
 * 
 */
class TrainingShmWriter {
public:
    /**
     * @brief Constructs the writer and creates the shared memory segment.
     * @param max_capacity Ring buffer capacity in positions (default: 20,000)
     * @param segment_name Shared memory segment name (default: "/az_training")
     * @throws std::runtime_error if segment creation, sizing, or mapping fails
     */
    explicit TrainingShmWriter(size_t max_capacity = 20000,
                               const std::string& segment_name = "/az_training");
    
    /**
     * @brief Destructor - unmaps and unlinks the shared memory segment.
     */
    ~TrainingShmWriter();
    
    /**
     * @brief Flushes a completed game to shared memory.
     * @param pool Position pool containing the game data with finalized z-values
     * 
     * Thread-safe: Uses internal mutex protection.
     * This should be called after PositionPool::finalize_game() has assigned outcomes.
     */
    void flush_game(const PositionPool& pool);
    
    /**
     * @brief Gets the buffer capacity.
     * @return Maximum number of positions the buffer can hold
     */
    size_t max_capacity() const {
        return max_capacity_;
    }
    
    /**
     * @brief Signals shutdown to the Python training process.
     */
    void shutdown();
    
    /**
     * @brief Checks if shutdown has been requested.
     * @return true if shutdown signal is set
     */
    bool is_shutdown() const;
    
    /**
     * @brief Gets the current generation counter.
     * @return Number of games flushed to the buffer
     */
    uint64_t generation() const;
    
    /**
     * @brief Gets the current number of positions in the buffer.
     * @return Current buffer size (capped at max_capacity)
     */
    uint32_t current_size() const;
    
    /**
     * @brief Gets the current write index in the ring buffer.
     * @return Index where next write will occur
     */
    uint32_t write_index() const;

private:
    TrainingShmWriter(const TrainingShmWriter&) = delete;
    TrainingShmWriter& operator=(const TrainingShmWriter&) = delete;
    
    std::string segment_name_;          ///< Shared memory segment name
    size_t max_capacity_;               ///< Maximum buffer capacity in positions
    int shm_fd_;                        ///< Shared memory file descriptor
    void* shm_base_;                    ///< Base pointer to mapped memory
    size_t shm_size_;                   ///< Total size of shared memory segment
    
    TrainingBufferHeader* header_;      ///< Pointer to buffer header
    TrainingPosition* positions_;       ///< Pointer to positions array
    
    std::mutex write_mutex_;            ///< Mutex protecting concurrent writes
};