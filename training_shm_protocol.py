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

TRAINING_BOARD_SIZE = 48      # 3 × 4 × 4
TRAINING_POLICY_SIZE = 16     # 4 × 4
TRAINING_POSITION_BYTES = 384

# ============================================================================
# CTYPES STRUCTURES
# ============================================================================

class TrainingPosition(ctypes.Structure):
    """
    Maps to C++ TrainingPosition struct (384 bytes)
    """
    _fields_ = [
        ("board", ctypes.c_float * TRAINING_BOARD_SIZE),     # 192 bytes at offset 0
        ("pi", ctypes.c_float * TRAINING_POLICY_SIZE),       # 64 bytes at offset 192
        ("z", ctypes.c_float),                               # 4 bytes at offset 256
        ("mask", ctypes.c_float * TRAINING_POLICY_SIZE),     # 64 bytes at offset 260
        ("_padding", ctypes.c_uint8 * 60),                   # 60 bytes at offset 324
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
    expected_size = TRAINING_POSITION_BYTES
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
# NUMPY VIEW CREATION
# ============================================================================

def create_numpy_views(shm_buffer: memoryview, max_capacity: int) -> Dict[str, np.ndarray]:
    """
    Create NumPy views into the shared memory buffer.
    
    Args:
        shm_buffer: memoryview of the shared memory segment
        max_capacity: ring buffer capacity (number of positions)
    
    Returns:
        Dictionary of NumPy arrays backed by shared memory:
        - 'boards': [max_capacity, 3, 4, 4] float32
        - 'pi': [max_capacity, 16] float32
        - 'z': [max_capacity] float32
        - 'mask': [max_capacity, 16] float32
    """
    # Calculate offsets
    header_size = ctypes.sizeof(TrainingBufferHeader)
    position_size = TRAINING_POSITION_BYTES
    
    # Create a flat float32 array view of all positions
    positions_offset = header_size
    positions_size = max_capacity * position_size
    
    # Total floats in the position data
    floats_per_position = (TRAINING_BOARD_SIZE + TRAINING_POLICY_SIZE + 1 + 
                           TRAINING_POLICY_SIZE)  # board + pi + z + mask = 48 + 16 + 1 + 16 = 81
    
    # But we have padding, so work with bytes instead
    positions_bytes = np.frombuffer(
        shm_buffer, 
        dtype=np.uint8, 
        count=positions_size, 
        offset=positions_offset
    )
    
    # Create structured views
    boards_list = []
    pi_list = []
    z_list = []
    mask_list = []
    
    for i in range(max_capacity):
        pos_offset = i * position_size
        
        # Board: 192 bytes (48 floats) at offset 0
        board = np.frombuffer(
            positions_bytes,
            dtype=np.float32,
            count=TRAINING_BOARD_SIZE,
            offset=pos_offset
        ).reshape(3, 4, 4)
        boards_list.append(board)
        
        # Pi: 64 bytes (16 floats) at offset 192
        pi = np.frombuffer(
            positions_bytes,
            dtype=np.float32,
            count=TRAINING_POLICY_SIZE,
            offset=pos_offset + 192
        )
        pi_list.append(pi)
        
        # Z: 4 bytes (1 float) at offset 256
        z = np.frombuffer(
            positions_bytes,
            dtype=np.float32,
            count=1,
            offset=pos_offset + 256
        )[0]
        z_list.append(z)
        
        # Mask: 64 bytes (16 floats) at offset 260
        mask = np.frombuffer(
            positions_bytes,
            dtype=np.float32,
            count=TRAINING_POLICY_SIZE,
            offset=pos_offset + 260
        )
        mask_list.append(mask)
    
    # Stack into arrays (these are views, not copies!)
    views = {
        'boards': np.array(boards_list),      # [max_capacity, 3, 4, 4]
        'pi': np.array(pi_list),              # [max_capacity, 16]
        'z': np.array(z_list),                # [max_capacity]
        'mask': np.array(mask_list),          # [max_capacity, 16]
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