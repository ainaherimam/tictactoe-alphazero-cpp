#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>

// ============================================================================
// CONFIGURATION CONSTANTS (must match Python exactly)
// ============================================================================

constexpr size_t MAX_BATCH_SIZE = 256;
constexpr size_t INPUT_CHANNELS = 3;
constexpr size_t BOARD_HEIGHT = 4;
constexpr size_t BOARD_WIDTH = 4;
constexpr size_t INPUT_SIZE = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;  // 48
constexpr size_t POLICY_SIZE = BOARD_HEIGHT * BOARD_WIDTH;                   // 16

// ============================================================================
// JOB STATE MACHINE (must match Python JobState enum)
// ============================================================================

enum class JobState : uint8_t {
    FREE = 0,        // Slot is available
    WRITING = 1,     // C++ is writing request data
    READY = 2,       // Request is ready for Python to process
    PROCESSING = 3,  // Python is processing this request
    DONE = 4         // Python has written the response
};

// ============================================================================
// REQUEST STRUCTURE
// ============================================================================

struct EvalRequest {
    // IMPORTANT: Field order must match Python exactly!
    std::atomic<uint64_t> job_id{0};           // 8 bytes at offset 0
    std::atomic<JobState> state{JobState::FREE}; // 1 byte at offset 8
    uint8_t _padding1[7];                      // 7 bytes padding to align to 16
    
    // Input data (written by C++)
    float board_state[INPUT_SIZE];             // 192 bytes (48 * 4) at offset 16
    float legal_mask[POLICY_SIZE];             // 64 bytes (16 * 4) at offset 208
    
    // Padding to reach 320 bytes total (cache-line aligned)
    uint8_t _padding2[48];                     // 48 bytes padding
    
    // Total: 8 + 1 + 7 + 192 + 64 + 48 = 320 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(EvalRequest) == 320, "EvalRequest must be 320 bytes to match Python");
static_assert(offsetof(EvalRequest, job_id) == 0, "job_id must be at offset 0");
static_assert(offsetof(EvalRequest, state) == 8, "state must be at offset 8");
static_assert(offsetof(EvalRequest, board_state) == 16, "board_state must be at offset 16");

// ============================================================================
// RESPONSE STRUCTURE
// ============================================================================

struct EvalResponse {
    // IMPORTANT: Field order must match Python exactly!
    std::atomic<uint64_t> job_id{0};           // 8 bytes at offset 0
    
    
    // Output data (written by Python)
    float policy[POLICY_SIZE];                 // 64 bytes (16 * 4) at offset 8
    float value;                               // 4 bytes at offset 72
    
    // Padding to reach 128 bytes total (cache-line aligned)
    uint8_t _padding[52];                      // 52 bytes padding
    
    // Total: 8 + 64 + 4 + 52 = 128 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(EvalResponse) == 128, "EvalResponse must be 128 bytes to match Python");
static_assert(offsetof(EvalResponse, job_id) == 0, "job_id must be at offset 0");
static_assert(offsetof(EvalResponse, policy) == 8, "policy must be at offset 8");

// ============================================================================
// SHARED MEMORY BUFFER
// ============================================================================

struct SharedMemoryBuffer {
    // Control section (must match Python field order exactly!)
    std::atomic<bool> server_ready{false};              // 1 byte at offset 0
    std::atomic<bool> shutdown_requested{false};        // 1 byte at offset 1
    uint8_t _padding1[6];                               // 6 bytes padding to align to 8
    
    std::atomic<uint64_t> total_requests_submitted{0};  // 8 bytes at offset 8
    std::atomic<uint64_t> total_requests_completed{0};  // 8 bytes at offset 16
    std::atomic<uint64_t> total_batches_processed{0};   // 8 bytes at offset 24
    std::atomic<uint32_t> next_slot_hint{0};            // 4 bytes at offset 32
    
    
    uint8_t _padding2[28];                              // 28 bytes padding to reach 64
    
    // Request and response arrays (starting at offset 64)
    EvalRequest requests[MAX_BATCH_SIZE];
    EvalResponse responses[MAX_BATCH_SIZE];
    
} __attribute__((aligned(64)));

static_assert(offsetof(SharedMemoryBuffer, server_ready) == 0, "server_ready at offset 0");
static_assert(offsetof(SharedMemoryBuffer, shutdown_requested) == 1, "shutdown_requested at offset 1");
static_assert(offsetof(SharedMemoryBuffer, total_requests_submitted) == 8, "total_requests_submitted at offset 8");
static_assert(offsetof(SharedMemoryBuffer, requests) == 64, "requests must start at offset 64");

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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