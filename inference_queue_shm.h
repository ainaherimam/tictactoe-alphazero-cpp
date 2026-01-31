#pragma once

#include "shared_memory_protocol.h"
#include <string>
#include <chrono>
#include <stdexcept>
#include <memory>

/**
 * SharedMemoryInferenceQueue
 * 
 * Low-level IPC primitive for communication with Python inference server.
 * 
 * Responsibilities:
 * - Attach to shared memory
 * - Allocate slots for requests
 * - Submit inference jobs (raw float arrays)
 * - Wait for responses
 * - NO framework dependencies (no Torch, no MCTS logic)
 * 
 * Thread-safe: Multiple threads can submit requests concurrently.
 */
class SharedMemoryInferenceQueue {
public:
    /**
     * Constructor
     * @param shm_name: Name of shared memory segment (e.g., "/mcts_jax_inference")
     */
    explicit SharedMemoryInferenceQueue(const std::string& shm_name);
    
    /**
     * Destructor - cleanup
     */
    ~SharedMemoryInferenceQueue();
    
    // Delete copy and move to avoid double-free
    SharedMemoryInferenceQueue(const SharedMemoryInferenceQueue&) = delete;
    SharedMemoryInferenceQueue& operator=(const SharedMemoryInferenceQueue&) = delete;
    SharedMemoryInferenceQueue(SharedMemoryInferenceQueue&&) = delete;
    SharedMemoryInferenceQueue& operator=(SharedMemoryInferenceQueue&&) = delete;
    
    // ========================================================================
    // CORE API
    // ========================================================================
    
    /**
     * Submit an inference request (non-blocking)
     * 
     * @param board_state: Flattened board [INPUT_SIZE]
     * @param legal_mask: Legal moves mask [POLICY_SIZE]
     * @return job_id: Unique identifier for this job
     * 
     * This will spin-wait if all slots are full.
     */
    uint64_t submit(const float* board_state, const float* legal_mask);
    
    /**
     * Check if a job is done (non-blocking)
     * 
     * @param job_id: Job identifier
     * @return true if response is ready
     */
    bool is_done(uint64_t job_id);
    
    /**
     * Get response for a completed job (non-blocking)
     * 
     * @param job_id: Job identifier
     * @param out_policy: Output buffer [POLICY_SIZE]
     * @param out_value: Output value
     * @return true if response was retrieved, false if not ready
     * 
     * Precondition: is_done(job_id) must be true
     * After successful call, the slot is freed for reuse.
     */
    bool get_response(uint64_t job_id, 
                     float* out_policy, 
                     float* out_value);
    
    /**
     * Wait for a job to complete (blocking)
     * 
     * @param job_id: Job identifier
     * @param out_policy: Output buffer [POLICY_SIZE]
     * @param out_value: Output value
     * @param timeout_ms: Max time to wait (default: 10 seconds)
     * 
     * Throws std::runtime_error on timeout
     * After successful call, the slot is freed for reuse.
     */
    void wait(uint64_t job_id,
              float* out_policy,
              float* out_value,
              int timeout_ms = 10000);
    
    // ========================================================================
    // STATUS API
    // ========================================================================
    
    /**
     * Check if Python server is ready
     */
    bool is_server_ready() const;
    
    /**
     * Request graceful shutdown
     */
    void request_shutdown();
    
    /**
     * Count pending requests (for monitoring)
     */
    size_t count_pending() const;
    
    /**
     * Get statistics
     */
    struct Stats {
        uint64_t submitted;
        uint64_t completed;
        uint64_t batches;
        size_t pending;
    };
    Stats get_stats() const;
    
    
private:
    // Shared memory
    std::string shm_name_;
    int shm_fd_;
    SharedMemoryBuffer* buffer_;
    
    // Job ID generator (thread-safe)
    std::atomic<uint64_t> next_job_id_;
    
    // Slot allocation
    int allocate_slot();
    int find_slot_for_job(uint64_t job_id) const;
    
    // Helpers
    void verify_buffer_sizes() const;
    void dump_shared_memory_state() const;
};