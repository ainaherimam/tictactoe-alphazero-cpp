#pragma once

#include "shared_memory_protocol.h"
#include <string>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <condition_variable>

/**
 * @class SharedMemoryInferenceQueue
 * @brief Low-level IPC primitive for communication with JAX inference server.
 * 
 * This class provides a thread-safe interface for submitting inference requests
 * to a JAX neural network server via shared memory. It handles slot
 * allocation, request submission, and response retrieval.
 * 
 * Thread-safe: Multiple threads can submit requests concurrently.
 */
class SharedMemoryInferenceQueue {
public:
    /**
     * @brief Constructs and connects to the shared memory segment.
     * @param shm_name Name of shared memory segment (e.g., "/mcts_jax_inference")
     * @throws std::runtime_error if shared memory cannot be opened or mapped
     */
    explicit SharedMemoryInferenceQueue(const std::string& shm_name);
    
    /**
     * @brief Destructor - unmaps shared memory and closes file descriptor.
     */
    ~SharedMemoryInferenceQueue();
    
    SharedMemoryInferenceQueue(const SharedMemoryInferenceQueue&) = delete;
    SharedMemoryInferenceQueue& operator=(const SharedMemoryInferenceQueue&) = delete;
    SharedMemoryInferenceQueue(SharedMemoryInferenceQueue&&) = delete;
    SharedMemoryInferenceQueue& operator=(SharedMemoryInferenceQueue&&) = delete;
    
    /**
     * @brief Submits an inference request (non-blocking once slot is acquired).
     * @param board_state Flattened board state array [INPUT_SIZE]
     * @param legal_mask Legal moves mask array [POLICY_SIZE]
     * @return Unique job identifier for this request
     * @throws std::invalid_argument if pointers are null
     * 
     * This will spin-wait if all slots are full until one becomes available.
     */
    uint64_t submit(const float* board_state, const float* legal_mask);
    
    /**
     * @brief Checks if a job is complete (non-blocking).
     * @param job_id Job identifier returned by submit()
     * @return true if response is ready, false otherwise
     */
    bool is_done(uint64_t job_id);
    
    /**
     * @brief Retrieves response for a completed job (non-blocking).
     * @param job_id Job identifier returned by submit()
     * @param out_policy Output buffer for policy [POLICY_SIZE]
     * @param out_value Output pointer for value
     * @return true if response was retrieved, false if not ready
     * @throws std::invalid_argument if pointers are null
     * @throws std::runtime_error if job ID mismatch occurs
     * 
     * Precondition: is_done(job_id) should be true.
     * After successful call, the slot is freed for reuse.
     */
    bool get_response(uint64_t job_id, 
                      float* out_policy, 
                      float* out_value);
    
    /**
     * @brief Waits for a job to complete (blocking).
     * @param job_id Job identifier returned by submit()
     * @param out_policy Output buffer for policy [POLICY_SIZE]
     * @param out_value Output pointer for value
     * @param timeout_ms Maximum time to wait in milliseconds (default: 10 seconds)
     * @throws std::runtime_error on timeout
     * @throws std::invalid_argument if pointers are null
     * 
     * After successful call, the slot is freed for reuse.
     */
    void wait(uint64_t job_id,
              float* out_policy,
              float* out_value,
              int timeout_ms = 1000);
    
    /**
     * @brief Checks if the Python inference server is ready.
     * @return true if server is ready to process requests
     */
    bool is_server_ready() const;
    
    /**
     * @brief Requests graceful shutdown of the inference server.
     */
    void request_shutdown();
    
    /**
     * @brief Counts pending requests across all slots.
     * @return Number of requests not yet completed
     */
    size_t count_pending() const;
    
    /**
     * @struct Stats
     * @brief Statistics about inference queue activity.
     */
    struct Stats {
        uint64_t submitted;  ///< Total requests submitted
        uint64_t completed;  ///< Total requests completed
        uint64_t batches;    ///< Total batches processed by server
        size_t pending;      ///< Current pending requests
    };
    
    /**
     * @brief Retrieves current statistics.
     * @return Stats structure with current counts
     */
    Stats get_stats() const;

    /**
     * @brief Notifies waiting threads that a batch has completed.
     */
    void notify_batch_complete();

    /**
     * @brief Waits for the Python server to become ready.
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if server became ready, false on timeout
     */
    bool wait_for_server(int timeout_ms);
    
private:
    std::string shm_name_;              ///< Name of shared memory segment
    int shm_fd_;                        ///< Shared memory file descriptor
    SharedMemoryBuffer* buffer_;        ///< Pointer to mapped shared memory
    
    std::atomic<uint64_t> next_job_id_; ///< Thread-safe job ID generator

    std::mutex wait_mutex_;             ///< Mutex for condition variable
    std::condition_variable wait_cv_;   ///< Condition variable for batch notifications
    uint32_t last_seen_notification_;   ///< Last seen notification counter

    /**
     * @brief Allocates a free slot for a new request.
     * @return Slot index (0 to MAX_BATCH_SIZE-1)
     * 
     * Spins until a slot becomes available.
     */
    int allocate_slot();
    
    /**
     * @brief Finds the slot containing a specific job.
     * @param job_id Job identifier to search for
     * @return Slot index if found, -1 otherwise
     */
    int find_slot_for_job(uint64_t job_id) const;
    
    /**
     * @brief Verifies and prints shared memory buffer sizes.
     */
    void verify_buffer_sizes() const;
    
    /**
     * @brief Dumps the current state of shared memory for debugging.
     */
    void dump_shared_memory_state() const;
};