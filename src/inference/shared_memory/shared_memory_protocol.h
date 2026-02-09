#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "core/game/constants.h"

/**
 * @file shared_memory_protocol.h
 * @brief Defines the shared memory protocol between C++ and Python inference server.
 * 
 * CRITICAL: All structures, field orders, and sizes must match the Python implementation
 * exactly to ensure correct data layout across the language boundary.
 */

/**
 * @brief Maximum number of concurrent inference requests.
 * 
 * Must match Python configuration exactly.
 */
constexpr size_t MAX_BATCH_SIZE = 256;

/**
 * @enum JobState
 * @brief State machine for tracking inference request lifecycle.
 * 
 * Must match Python JobState enum exactly.
 */
enum class JobState : uint8_t {
    FREE = 0,        ///< Slot is available for new request
    WRITING = 1,     ///< C++ is writing request data
    READY = 2,       ///< Request is ready for Python to process
    PROCESSING = 3,  ///< Python is processing this request
    DONE = 4         ///< Python has written the response
};

/**
 * @struct EvalRequest
 * @brief Request structure for neural network inference.
 * 
 * IMPORTANT: Field order and padding must match Python exactly!
 * Total size: 320 bytes (cache-line aligned)
 */
struct EvalRequest {
    std::atomic<uint64_t> job_id{0};           ///< Unique job identifier (8 bytes at offset 0)
    std::atomic<JobState> state{JobState::FREE}; ///< Current job state (1 byte at offset 8)
    uint8_t _padding1[7];                      ///< Padding to align to 16 bytes
    
    float board_state[INPUT_SIZE];             ///< Board state representation (192 bytes at offset 16)
    float legal_mask[POLICY_SIZE];             ///< Legal move mask (64 bytes at offset 208)
    
    uint8_t _padding2[48];                     ///< Padding to reach 320 bytes total
    
} __attribute__((aligned(64)));

static_assert(sizeof(EvalRequest) == 320, "EvalRequest must be 320 bytes to match Python");
static_assert(offsetof(EvalRequest, job_id) == 0, "job_id must be at offset 0");
static_assert(offsetof(EvalRequest, state) == 8, "state must be at offset 8");
static_assert(offsetof(EvalRequest, board_state) == 16, "board_state must be at offset 16");

/**
 * @struct EvalResponse
 * @brief Response structure containing neural network outputs.
 * 
 * IMPORTANT: Field order and padding must match Python exactly!
 * Total size: 128 bytes (cache-line aligned)
 */
struct EvalResponse {
    std::atomic<uint64_t> job_id{0};           ///< Job identifier matching the request (8 bytes at offset 0)
    
    float policy[POLICY_SIZE];                 ///< Policy distribution output (64 bytes at offset 8)
    float value;                               ///< Value estimate output (4 bytes at offset 72)
    
    uint8_t _padding[52];                      ///< Padding to reach 128 bytes total
    
} __attribute__((aligned(64)));

static_assert(sizeof(EvalResponse) == 128, "EvalResponse must be 128 bytes to match Python");
static_assert(offsetof(EvalResponse, job_id) == 0, "job_id must be at offset 0");
static_assert(offsetof(EvalResponse, policy) == 8, "policy must be at offset 8");

/**
 * @struct SharedMemoryBuffer
 * @brief Main shared memory buffer containing control data and request/response arrays.
 * 
 * IMPORTANT: Field order must match Python implementation exactly!
 * The buffer is designed for cache-line alignment and efficient concurrent access.
 */
struct SharedMemoryBuffer {
    std::atomic<bool> server_ready{false};              ///< Server initialization flag (1 byte at offset 0)
    std::atomic<bool> shutdown_requested{false};        ///< Shutdown signal flag (1 byte at offset 1)
    uint8_t _padding1[6];                               ///< Padding to align to 8 bytes
    
    std::atomic<uint64_t> total_requests_submitted{0};  ///< Total requests submitted (8 bytes at offset 8)
    std::atomic<uint64_t> total_requests_completed{0};  ///< Total requests completed (8 bytes at offset 16)
    std::atomic<uint64_t> total_batches_processed{0};   ///< Total batches processed (8 bytes at offset 24)
    std::atomic<uint32_t> next_slot_hint{0};            ///< Hint for next slot allocation (4 bytes at offset 32)
    std::atomic<uint32_t> notification_counter{0};      ///< Batch completion counter (4 bytes at offset 36)
    
    uint8_t _padding2[24];                              ///< Padding to reach 64 bytes
    
    EvalRequest requests[MAX_BATCH_SIZE];               ///< Request slots (starting at offset 64)
    EvalResponse responses[MAX_BATCH_SIZE];             ///< Response slots
    
} __attribute__((aligned(64)));

static_assert(offsetof(SharedMemoryBuffer, server_ready) == 0, "server_ready at offset 0");
static_assert(offsetof(SharedMemoryBuffer, shutdown_requested) == 1, "shutdown_requested at offset 1");
static_assert(offsetof(SharedMemoryBuffer, total_requests_submitted) == 8, "total_requests_submitted at offset 8");
static_assert(offsetof(SharedMemoryBuffer, requests) == 64, "requests must start at offset 64");

/**
 * @brief Converts JobState enum to human-readable string.
 * @param state The job state to convert
 * @return String representation of the state
 */
inline const char* job_state_to_string(JobState state) {
    switch (state) {
        case JobState::FREE: return "FREE";
        case JobState::WRITING: return "WRITING";
        case JobState::READY: return "READY";
        case JobState::PROCESSING: return "PROCESSING";
        case JobState::DONE: return "DONE";
        default: return "UNKNOWN";
    }
}