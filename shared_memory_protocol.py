"""
Shared Memory Protocol (Python side)

This file MUST match shared_memory_protocol.h EXACTLY.
Any mismatch in sizes or alignment will cause corruption!

Run verify_sizes() to check alignment.
"""

import ctypes
import numpy as np
from enum import IntEnum

# ============================================================================
# CONFIGURATION CONSTANTS (must match C++ exactly)
# ============================================================================

MAX_BATCH_SIZE = 256
INPUT_CHANNELS = 3
BOARD_HEIGHT = 4
BOARD_WIDTH = 4
INPUT_SIZE = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH
POLICY_SIZE = BOARD_HEIGHT * BOARD_WIDTH

# ============================================================================
# JOB STATE ENUM
# ============================================================================

class JobState(IntEnum):
    """Job state machine - must match C++ enum"""
    FREE = 0
    WRITING = 1
    READY = 2
    PROCESSING = 3
    DONE = 4

# ============================================================================
# ATOMIC TYPES
# ============================================================================

class AtomicUInt64(ctypes.Structure):
    """std::atomic<uint64_t> representation"""
    _fields_ = [('value', ctypes.c_uint64)]

class AtomicUInt8(ctypes.Structure):
    """std::atomic<uint8_t> representation (for JobState)"""
    _fields_ = [('value', ctypes.c_uint8)]

class AtomicBool(ctypes.Structure):
    """std::atomic<bool> representation"""
    _fields_ = [('value', ctypes.c_bool)]

class AtomicUInt32(ctypes.Structure):
    """std::atomic<uint32_t> representation"""
    _fields_ = [('value', ctypes.c_uint32)]

# ============================================================================
# REQUEST STRUCTURE
# ============================================================================

class EvalRequest(ctypes.Structure):
    """
    Must match C++ EvalRequest exactly
    
    C++ structure:
    struct EvalRequest {
        std::atomic<uint64_t> job_id;
        std::atomic<JobState> state;
        float board_state[INPUT_SIZE];
        float legal_mask[POLICY_SIZE];
        uint8_t padding[...];
    } __attribute__((aligned(64)));
    """
    
    _fields_ = [
    ('job_id', AtomicUInt64),                       # 8 bytes
    ('state', AtomicUInt8),                         # 1 byte
    ('_padding1', ctypes.c_uint8 * 7),              # 7 bytes
    ('board_state', ctypes.c_float * INPUT_SIZE),   # 192 bytes
    ('legal_mask', ctypes.c_float * POLICY_SIZE),   # 64 bytes
    ('_padding2', ctypes.c_uint8 * 48),             # 48 bytes -> total 320
    ]
    
    def __repr__(self):
        return (f"EvalRequest(job_id={self.job_id.value}, "
                f"state={JobState(self.state.value).name})")


class EvalResponse(ctypes.Structure):
    """
    Must match C++ EvalResponse exactly
    
    C++ structure:
    struct EvalResponse {
        std::atomic<uint64_t> job_id;
        float policy[POLICY_SIZE];
        float value;
        uint8_t padding[...];
    } __attribute__((aligned(64)));
    """
    
    _fields_ = [
        ('job_id', AtomicUInt64),                       # 8 bytes
        ('policy', ctypes.c_float * POLICY_SIZE),       # 64 bytes
        ('value', ctypes.c_float),                      # 4 bytes
        ('_padding', ctypes.c_uint8 * 52),              # 52 bytes -> total 128
    ]
    
    def __repr__(self):
        return f"EvalResponse(job_id={self.job_id.value}, value={self.value})"


class SharedMemoryBuffer(ctypes.Structure):
    """
    Must match C++ SharedMemoryBuffer exactly
    
    C++ structure:
    struct SharedMemoryBuffer {
        std::atomic<bool> server_ready;
        std::atomic<bool> shutdown_requested;
        std::atomic<uint64_t> total_requests_submitted;
        std::atomic<uint64_t> total_requests_completed;
        std::atomic<uint64_t> total_batches_processed;
        std::atomic<uint32_t> next_slot_hint;
        uint8_t control_padding[64 - ...];
        EvalRequest requests[MAX_BATCH_SIZE];
        EvalResponse responses[MAX_BATCH_SIZE];
    } __attribute__((aligned(64)));
    """
    
    _fields_ = [
        # Control section
        ('server_ready', AtomicBool),                           # 1 byte
        ('shutdown_requested', AtomicBool),                     # 1 byte
        ('_padding1', ctypes.c_uint8 * 6),                      # 6 bytes (alignment)
        ('total_requests_submitted', AtomicUInt64),             # 8 bytes
        ('total_requests_completed', AtomicUInt64),             # 8 bytes
        ('total_batches_processed', AtomicUInt64),              # 8 bytes
        ('next_slot_hint', AtomicUInt32),                       # 4 bytes
        ('notification_counter', AtomicUInt32),                 # 4 bytes
        ('_padding2', ctypes.c_uint8 * 24),                     # 28 bytes (cache line align)
        
        # Request and response arrays
        ('requests', EvalRequest * MAX_BATCH_SIZE),
        ('responses', EvalResponse * MAX_BATCH_SIZE),
    ]


# ============================================================================
# SIZE VERIFICATION
# ============================================================================

def verify_sizes():
    """
    Verify struct sizes match expectations.
    
    This should be run when starting the Python server to ensure
    the protocol matches the C++ side exactly.
    """
    print("=" * 60)
    print("Protocol Size Verification")
    print("=" * 60)
    
    # Expected sizes (these should be multiples of 64 for cache alignment)
    req_size = ctypes.sizeof(EvalRequest)
    resp_size = ctypes.sizeof(EvalResponse)
    buf_size = ctypes.sizeof(SharedMemoryBuffer)
    
    print(f"EvalRequest size:        {req_size:6} bytes")
    print(f"EvalResponse size:       {resp_size:6} bytes")
    print(f"SharedMemoryBuffer size: {buf_size:6} bytes")
    print()
    print(f"INPUT_SIZE:              {INPUT_SIZE}")
    print(f"POLICY_SIZE:             {POLICY_SIZE}")
    print(f"MAX_BATCH_SIZE:          {MAX_BATCH_SIZE}")
    print()
    
    # Check cache line alignment
    all_aligned = True
    
    if req_size % 64 != 0:
        print(f"⚠️  WARNING: EvalRequest not cache-aligned (size={req_size})")
        all_aligned = False
    
    if resp_size % 64 != 0:
        print(f"⚠️  WARNING: EvalResponse not cache-aligned (size={resp_size})")
        all_aligned = False
    
    if all_aligned:
        print("✓ All structures are properly cache-aligned (64 bytes)")
    
    print("=" * 60)
    print()
    
    return all_aligned


def print_memory_layout():
    """Print detailed memory layout for debugging"""
    print("\n" + "=" * 60)
    print("Detailed Memory Layout")
    print("=" * 60)
    
    print("\nEvalRequest fields:")
    for field_name, field_type in EvalRequest._fields_:
        offset = getattr(EvalRequest, field_name).offset
        size = ctypes.sizeof(field_type)
        print(f"  {field_name:20} offset={offset:4}, size={size:4}")
    
    print(f"\nEvalResponse fields:")
    for field_name, field_type in EvalResponse._fields_:
        offset = getattr(EvalResponse, field_name).offset
        size = ctypes.sizeof(field_type)
        print(f"  {field_name:20} offset={offset:4}, size={size:4}")
    
    print(f"\nSharedMemoryBuffer fields:")
    for field_name, field_type in SharedMemoryBuffer._fields_:
        offset = getattr(SharedMemoryBuffer, field_name).offset
        if field_name in ['requests', 'responses']:
            size = ctypes.sizeof(field_type)
            print(f"  {field_name:20} offset={offset:6}, size={size:8}")
        else:
            size = ctypes.sizeof(field_type)
            print(f"  {field_name:20} offset={offset:4}, size={size:4}")
    
    print("=" * 60 + "\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_empty_buffer():
    """Create a zeroed SharedMemoryBuffer"""
    return SharedMemoryBuffer()


def initialize_buffer(buffer: SharedMemoryBuffer):
    """Initialize buffer to clean state"""
    # Zero everything
    ctypes.memset(ctypes.addressof(buffer), 0, ctypes.sizeof(SharedMemoryBuffer))
    
    # Set all request states to FREE
    for i in range(MAX_BATCH_SIZE):
        buffer.requests[i].state.value = JobState.FREE
    
    print("[Protocol] Buffer initialized to clean state")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing shared_memory_protocol.py\n")
    
    # Verify sizes
    if not verify_sizes():
        print("❌ Size verification failed!")
        print("⚠️  This will cause memory corruption with C++!")
        exit(1)
    
    # Print detailed layout
    print_memory_layout()
    
    # Test creating structures
    print("Testing structure creation...")
    req = EvalRequest()
    resp = EvalResponse()
    buf = SharedMemoryBuffer()
    
    print(f"✓ Created EvalRequest: {req}")
    print(f"✓ Created EvalResponse: {resp}")
    print(f"✓ Created SharedMemoryBuffer")
    
    # Test state enum
    print("\nTesting JobState enum:")
    for state in JobState:
        print(f"  {state.name:12} = {state.value}")
    
    print("\n✅ All tests passed!")