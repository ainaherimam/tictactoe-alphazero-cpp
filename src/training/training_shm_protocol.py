"""
Training Shared Memory Protocol - Python Side
==============================================
Mirrors the C++ training_shm_protocol.h structs using ctypes.
Provides verification and NumPy view creation.

Segment: /az_training
Layout: [64-byte header][N × TRAINING_POSITION_BYTES positions]
"""

import ctypes
import numpy as np
from typing import Dict, Tuple

from src.constants import INPUT_SIZE, POLICY_SIZE, INPUT_PLANES, BOARD_HEIGHT, BOARD_WIDTH

# ============================================================================
# CONSTANTS (must match C++ exactly)
# ============================================================================

TRAINING_BOARD_SIZE  = INPUT_SIZE    # INPUT_PLANES × BOARD_HEIGHT × BOARD_WIDTH floats
TRAINING_POLICY_SIZE = POLICY_SIZE   # BOARD_CELLS floats

# Derived: total data bytes in one TrainingPosition (board + pi + z + mask).
_TRAINING_DATA_BYTES     = (TRAINING_BOARD_SIZE + TRAINING_POLICY_SIZE) * 4 + 4 + TRAINING_POLICY_SIZE * 4
# Round up to the next 64-byte cache-line boundary.
TRAINING_POSITION_BYTES  = ((_TRAINING_DATA_BYTES + 63) // 64) * 64
_TRAINING_POSITION_PAD   = TRAINING_POSITION_BYTES - _TRAINING_DATA_BYTES

# Field byte offsets (used for NumPy view strides and verification).
_OFFSET_BOARD = 0
_OFFSET_PI    = TRAINING_BOARD_SIZE * 4
_OFFSET_Z     = _OFFSET_PI + TRAINING_POLICY_SIZE * 4
_OFFSET_MASK  = _OFFSET_Z + 4

# ============================================================================
# CTYPES STRUCTURES
# ============================================================================

class TrainingPosition(ctypes.Structure):
    """
    Maps to C++ TrainingPosition struct (TRAINING_POSITION_BYTES, cache-line aligned).

    Matches training_shm_protocol.h exactly:
        float board[TRAINING_BOARD_SIZE]    TRAINING_BOARD_SIZE*4 bytes  @ offset _OFFSET_BOARD
        float pi[TRAINING_POLICY_SIZE]      TRAINING_POLICY_SIZE*4 bytes @ offset _OFFSET_PI
        float z                             4 bytes                       @ offset _OFFSET_Z
        float mask[TRAINING_POLICY_SIZE]    TRAINING_POLICY_SIZE*4 bytes @ offset _OFFSET_MASK
        uint8_t _padding[_TRAINING_POSITION_PAD]                          ← pads to cache line
    """
    _fields_ = [
        ("board",    ctypes.c_float * TRAINING_BOARD_SIZE),              # @ offset _OFFSET_BOARD
        ("pi",       ctypes.c_float * TRAINING_POLICY_SIZE),             # @ offset _OFFSET_PI
        ("z",        ctypes.c_float),                                     # @ offset _OFFSET_Z
        ("mask",     ctypes.c_float * TRAINING_POLICY_SIZE),             # @ offset _OFFSET_MASK
        ("_padding", ctypes.c_uint8 * _TRAINING_POSITION_PAD),           # padding to cache-line
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
    expected_size = TRAINING_POSITION_BYTES  # 260
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
        "board": _OFFSET_BOARD,
        "pi":    _OFFSET_PI,
        "z":     _OFFSET_Z,
        "mask":  _OFFSET_MASK,
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

    Each TrainingPosition is 320 bytes (60-byte padding for cache-line alignment).
    We use NumPy's stride mechanism to reach each data field directly without
    copying — the padding bytes are simply skipped by the larger stride.

    Args:
        shm_buffer: memoryview of the shared memory segment
        max_capacity: ring buffer capacity (number of positions)

    Returns:
        Dictionary of NumPy arrays backed by shared memory:
        - 'boards': [max_capacity, INPUT_PLANES, BOARD_HEIGHT, BOARD_WIDTH] float32
        - 'pi':     [max_capacity, POLICY_SIZE]                              float32
        - 'z':      [max_capacity]                                           float32
        - 'mask':   [max_capacity, POLICY_SIZE]                              float32
    """
    header_size   = ctypes.sizeof(TrainingBufferHeader)
    position_size = TRAINING_POSITION_BYTES

    positions_offset = header_size

    # BOARDS: shape=[max_capacity, INPUT_PLANES, BOARD_HEIGHT, BOARD_WIDTH]
    # Strides: position_size, BOARD_HEIGHT*BOARD_WIDTH*4, BOARD_WIDTH*4, 4
    _board_channel_stride = BOARD_HEIGHT * BOARD_WIDTH * 4
    _board_row_stride     = BOARD_WIDTH * 4
    boards = np.ndarray(
        shape=(max_capacity, INPUT_PLANES, BOARD_HEIGHT, BOARD_WIDTH),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + _OFFSET_BOARD,
        strides=(position_size, _board_channel_stride, _board_row_stride, 4),
    )

    # PI: shape=[max_capacity, POLICY_SIZE]
    pi = np.ndarray(
        shape=(max_capacity, TRAINING_POLICY_SIZE),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + _OFFSET_PI,
        strides=(position_size, 4),
    )

    # Z: shape=[max_capacity]
    z = np.ndarray(
        shape=(max_capacity,),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + _OFFSET_Z,
        strides=(position_size,),
    )

    # MASK: shape=[max_capacity, POLICY_SIZE]
    mask = np.ndarray(
        shape=(max_capacity, TRAINING_POLICY_SIZE),
        dtype=np.float32,
        buffer=shm_buffer,
        offset=positions_offset + _OFFSET_MASK,
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