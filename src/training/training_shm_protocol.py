"""
Training Shared Memory Protocol - Python Side
==============================================
Mirrors the C++ training_shm_protocol.h structs using ctypes.
Provides verification and NumPy view creation.

Segment: /az_training
Layout: [64-byte header][N × 384-byte positions]
"""

import ctypes
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# CONSTANTS (must match C++ exactly)
# ============================================================================

TRAINING_BOARD_SIZE = 48      # INPUT_SIZE  = INPUT_PLANES(3) × BOARD_HEIGHT(4) × BOARD_WIDTH(4)
TRAINING_POLICY_SIZE = 16     # POLICY_SIZE = BOARD_CELLS = 4 × 4

# C++ layout (training_shm_protocol.h):
#   float board[48]       = 192 bytes @ offset   0
#   float pi[16]          =  64 bytes @ offset 192
#   float z               =   4 bytes @ offset 256
#   float mask[16]        =  64 bytes @ offset 260
#   uint8_t _padding[60]  =  60 bytes @ offset 324  ← cache-line alignment to 384
# Total: 384 bytes, __attribute__((aligned(64)))
TRAINING_POSITION_BYTES = 384

# ============================================================================
# CTYPES STRUCTURES
# ============================================================================

class TrainingPosition(ctypes.Structure):
    """
    Maps to C++ TrainingPosition struct (384 bytes, cache-line aligned).

    Matches training_shm_protocol.h exactly:
        float board[48]      192 bytes  @ offset   0
        float pi[16]          64 bytes  @ offset 192
        float z                4 bytes  @ offset 256
        float mask[16]        64 bytes  @ offset 260
        uint8_t _padding[60]  60 bytes  @ offset 324  ← pads to 384 / cache line
    """
    _fields_ = [
        ("board",    ctypes.c_float * TRAINING_BOARD_SIZE),   # 192 bytes @ offset   0
        ("pi",       ctypes.c_float * TRAINING_POLICY_SIZE),  #  64 bytes @ offset 192
        ("z",        ctypes.c_float),                          #   4 bytes @ offset 256
        ("mask",     ctypes.c_float * TRAINING_POLICY_SIZE),  #  64 bytes @ offset 260
        ("_padding", ctypes.c_uint8 * 60),                    #  60 bytes @ offset 324
    ]


class TrainingBufferHeader(ctypes.Structure):
    """
    Maps to C++ TrainingBufferHeader struct (64 bytes)
    """
    _fields_ = [
        ("write_index", ctypes.c_uint32),      # 4 bytes at offset 0
        ("current_size", ctypes.c_uint32),     # 4 bytes at offset 4
        ("generation", ctypes.c_uint64),       # 8 bytes at offset 8
        ("shutdown", ctypes.c_bool),           # 1 byte at offset 16
        ("_padding", ctypes.c_uint8 * 47),     # 47 bytes at offset 17
    ]


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_sizes() -> bool:
    """
    Verify that Python structs match C++ layout exactly.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "="*70)
    print("TRAINING SHM PROTOCOL VERIFICATION")
    print("="*70)
    
    passed = True
    
    # Check TrainingPosition
    print("\n[TrainingPosition]")
    expected_size = TRAINING_POSITION_BYTES  # 324
    actual_size = ctypes.sizeof(TrainingPosition)
    
    print(f"  Expected size: {expected_size} bytes")
    print(f"  Actual size:   {actual_size} bytes")
    
    if actual_size != expected_size:
        print(f"  ❌ SIZE MISMATCH!")
        passed = False
    else:
        print(f"  ✅ Size correct")
    
    # Check field offsets
    offsets = {
        "board": 0,
        "pi": 192,
        "z": 256,
        "mask": 260,
    }
    
    for field_name, expected_offset in offsets.items():
        actual_offset = getattr(TrainingPosition, field_name).offset
        status = "✅" if actual_offset == expected_offset else "❌"
        print(f"  {status} {field_name:10} @ offset {actual_offset:3} (expected {expected_offset:3})")
        if actual_offset != expected_offset:
            passed = False
    
    # Check TrainingBufferHeader
    print("\n[TrainingBufferHeader]")
    expected_size = 64
    actual_size = ctypes.sizeof(TrainingBufferHeader)
    
    print(f"  Expected size: {expected_size} bytes")
    print(f"  Actual size:   {actual_size} bytes")
    
    if actual_size != expected_size:
        print(f"  ❌ SIZE MISMATCH!")
        passed = False
    else:
        print(f"  ✅ Size correct")
    
    # Check header field offsets
    header_offsets = {
        "write_index": 0,
        "current_size": 4,
        "generation": 8,
        "shutdown": 16,
    }
    
    for field_name, expected_offset in header_offsets.items():
        actual_offset = getattr(TrainingBufferHeader, field_name).offset
        status = "✅" if actual_offset == expected_offset else "❌"
        print(f"  {status} {field_name:15} @ offset {actual_offset:2} (expected {expected_offset:2})")
        if actual_offset != expected_offset:
            passed = False
    
    print("\n" + "="*70)
    if passed:
        print("✅ ALL CHECKS PASSED - Protocol verified!")
    else:
        print("❌ VERIFICATION FAILED - Fix struct layout!")
    print("="*70 + "\n")
    
    return passed


# ============================================================================
# NUMPY VIEW CREATION - FIXED VERSION
# ============================================================================

def create_numpy_views(shm_buffer: memoryview, max_capacity: int) -> Dict[str, np.ndarray]:
    """
    Create NumPy views into the shared memory buffer.

    CRITICAL: Returns VIEWS not copies - changes in shared memory are visible!

    Each TrainingPosition is 384 bytes (60-byte padding for cache-line alignment).
    We use NumPy's stride mechanism to reach each data field directly without
    copying — the padding bytes are simply skipped by the larger stride.

    Args:
        shm_buffer: memoryview of the shared memory segment
        max_capacity: ring buffer capacity (number of positions)

    Returns:
        Dictionary of NumPy arrays backed by shared memory:
        - 'boards': [max_capacity, 3, 4, 4] float32
        - 'pi':     [max_capacity, 16]       float32
        - 'z':      [max_capacity]            float32
        - 'mask':   [max_capacity, 16]       float32
    """
    header_size   = ctypes.sizeof(TrainingBufferHeader)   # 64
    position_size = TRAINING_POSITION_BYTES               # 324

    positions_offset = header_size
    
    # BOARDS: shape=[max_capacity, 3, 4, 4], field offset 0 within each position
    # Strides: position(324), channel(4*4*4=64), row(4*4=16), element(4)
    boards = np.ndarray(
        shape=(max_capacity, 3, 4, 4),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + 0,
        strides=(position_size, 64, 16, 4),
    )

    # PI: shape=[max_capacity, 16], field offset 192 within each position
    pi = np.ndarray(
        shape=(max_capacity, 16),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + 192,
        strides=(position_size, 4),
    )

    # Z: shape=[max_capacity], field offset 256 within each position
    z = np.ndarray(
        shape=(max_capacity,),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + 256,
        strides=(position_size,),
    )

    # MASK: shape=[max_capacity, 16], field offset 260 within each position
    mask = np.ndarray(
        shape=(max_capacity, 16),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + 260,
        strides=(position_size, 4),
    )
    
    views = {
        'boards': boards,
        'pi': pi,
        'z': z,
        'mask': mask,
    }
    
    return views


def get_segment_size(max_capacity: int) -> int:
    """Calculate total segment size for a given capacity."""
    return ctypes.sizeof(TrainingBufferHeader) + (max_capacity * TRAINING_POSITION_BYTES)


if __name__ == "__main__":
    # Run verification when module is executed
    verify_sizes()
    
    # Print segment sizes for common capacities
    print("\nSegment Sizes:")
    print("-" * 40)
    capacities = [20_000, 50_000, 100_000, 250_000, 500_000]
    for cap in capacities:
        size_mb = get_segment_size(cap) / (1024 * 1024)
        print(f"  {cap:7,} positions → {size_mb:7.1f} MB")