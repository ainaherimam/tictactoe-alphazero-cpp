#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "core/game/constants.h"

// ============================================================================
// TRAINING SHARED MEMORY PROTOCOL
// ============================================================================
// This is the training data buffer protocol - separate from inference SHM.
// C++ self-play workers write completed games here.
// Python JAX training process reads random samples.
//
// Segment name: /az_training
// Layout: [64-byte header] [N × TRAINING_POSITION_BYTES positions]
// ============================================================================

// ============================================================================
// CONFIGURATION CONSTANTS (must match Python exactly)
// ============================================================================

constexpr size_t TRAINING_BOARD_SIZE  = INPUT_SIZE;    // INPUT_PLANES * BOARD_HEIGHT * BOARD_WIDTH floats
constexpr size_t TRAINING_POLICY_SIZE = POLICY_SIZE;   // BOARD_CELLS floats

// Derived: total data bytes in one TrainingPosition (board + pi + z + mask).
constexpr size_t TRAINING_POSITION_DATA_BYTES =
    TRAINING_BOARD_SIZE  * sizeof(float) +  // board
    TRAINING_POLICY_SIZE * sizeof(float) +  // pi
    sizeof(float)                        +  // z
    TRAINING_POLICY_SIZE * sizeof(float);   // mask

// Round up to the next 64-byte cache-line boundary.
constexpr size_t TRAINING_POSITION_BYTES   = ((TRAINING_POSITION_DATA_BYTES + 63) / 64) * 64;
constexpr size_t TRAINING_POSITION_PADDING = TRAINING_POSITION_BYTES - TRAINING_POSITION_DATA_BYTES;

// ============================================================================
// TRAINING POSITION STRUCTURE (384 bytes, cache-line aligned)
// ============================================================================

struct TrainingPosition {
    // Board state: [3][4][4] float array (same as inference INPUT_SIZE)
    float board[TRAINING_BOARD_SIZE];          // 192 bytes at offset 0
    
    // Policy target: [16] float array (MCTS visit distribution)
    float pi[TRAINING_POLICY_SIZE];            // 64 bytes at offset 192
    
    // Value target: scalar (game outcome from this position's perspective)
    float z;                                   // 4 bytes at offset 256
    
    // Legal move mask: [16] float array (1.0 = legal, 0.0 = illegal)
    float mask[TRAINING_POLICY_SIZE];          // 64 bytes at offset 260
    
    // Padding to reach TRAINING_POSITION_BYTES (cache-line alignment)
    uint8_t _padding[TRAINING_POSITION_PADDING]; // TRAINING_POSITION_BYTES - TRAINING_POSITION_DATA_BYTES

    // Total: TRAINING_POSITION_BYTES (derived from TRAINING_BOARD_SIZE + TRAINING_POLICY_SIZE)
} __attribute__((aligned(64)));

static_assert(sizeof(TrainingPosition) == TRAINING_POSITION_BYTES, 
              "TrainingPosition must be 384 bytes");
static_assert(offsetof(TrainingPosition, board) == 0,
              "board must be at offset 0");
static_assert(offsetof(TrainingPosition, pi) == TRAINING_BOARD_SIZE * sizeof(float),
              "pi must immediately follow board");
static_assert(offsetof(TrainingPosition, z) == (TRAINING_BOARD_SIZE + TRAINING_POLICY_SIZE) * sizeof(float),
              "z must immediately follow pi");
static_assert(offsetof(TrainingPosition, mask) == (TRAINING_BOARD_SIZE + TRAINING_POLICY_SIZE) * sizeof(float) + sizeof(float),
              "mask must immediately follow z");

// ============================================================================
// TRAINING BUFFER HEADER (64 bytes, one cache line)
// ============================================================================

struct TrainingBufferHeader {
    // Ring buffer write position (wraps at max_capacity)
    std::atomic<uint32_t> write_index{0};      // 4 bytes at offset 0
    
    // Number of valid positions in buffer (caps at max_capacity)
    std::atomic<uint32_t> current_size{0};     // 4 bytes at offset 4
    
    // Incremented after each game flush - Python polls this
    std::atomic<uint64_t> generation{0};       // 8 bytes at offset 8
    
    // Shutdown signal (either side can set)
    std::atomic<bool> shutdown{false};         // 1 byte at offset 16
    
    // Padding to reach 64 bytes
    uint8_t _padding[47];                      // 47 bytes at offset 17
    
    // Total: 4 + 4 + 8 + 1 + 47 = 64 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(TrainingBufferHeader) == 64, 
              "TrainingBufferHeader must be 64 bytes");
static_assert(offsetof(TrainingBufferHeader, write_index) == 0, 
              "write_index must be at offset 0");
static_assert(offsetof(TrainingBufferHeader, current_size) == 4, 
              "current_size must be at offset 4");
static_assert(offsetof(TrainingBufferHeader, generation) == 8, 
              "generation must be at offset 8");
static_assert(offsetof(TrainingBufferHeader, shutdown) == 16, 
              "shutdown must be at offset 16");

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Calculate total segment size for a given capacity
inline size_t training_segment_size(size_t max_capacity) {
    return sizeof(TrainingBufferHeader) + (max_capacity * TRAINING_POSITION_BYTES);
}

// Get pointer to position array (after header)
inline TrainingPosition* training_positions_ptr(void* shm_base) {
    return reinterpret_cast<TrainingPosition*>(
        static_cast<char*>(shm_base) + sizeof(TrainingBufferHeader)
    );
}

// Get const pointer to position array
inline const TrainingPosition* training_positions_ptr(const void* shm_base) {
    return reinterpret_cast<const TrainingPosition*>(
        static_cast<const char*>(shm_base) + sizeof(TrainingBufferHeader)
    );
}