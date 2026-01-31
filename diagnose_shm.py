#!/usr/bin/env python3
"""
Diagnostic tool for Python side of shared memory inference queue.
This checks the shared memory state from Python's perspective.
"""

import numpy as np
import time
from multiprocessing import shared_memory
import struct

# Configuration (must match C++)
MAX_BATCH_SIZE = 256
INPUT_CHANNELS = 3
BOARD_HEIGHT = 4
BOARD_WIDTH = 4
INPUT_SIZE = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH  # 48
POLICY_SIZE = BOARD_HEIGHT * BOARD_WIDTH  # 16

# JobState enum values
class JobState:
    FREE = 0
    WRITING = 1
    READY = 2
    PROCESSING = 3
    DONE = 4
    
    @staticmethod
    def to_string(state):
        names = {0: "FREE", 1: "WRITING", 2: "READY", 3: "PROCESSING", 4: "DONE"}
        return names.get(state, f"UNKNOWN({state})")

def diagnose_shared_memory(shm_name="mcts_jax_inference"):
    """Diagnose the shared memory buffer from Python."""
    
    print(f"Diagnosing shared memory: /{shm_name}")
    print("=" * 60)
    
    try:
        # Attach to existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        print(f"✓ Successfully attached to shared memory")
        print(f"  Size: {len(shm.buf)} bytes")
        
    except FileNotFoundError:
        print(f"✗ Shared memory '/{shm_name}' not found!")
        print("  Is the Python inference server running?")
        return
    
    # Parse header (control data)
    offset = 0
    
    # bool server_ready
    server_ready = struct.unpack_from('?', shm.buf, offset)[0]
    offset += 1
    
    # bool shutdown_requested  
    shutdown_requested = struct.unpack_from('?', shm.buf, offset)[0]
    offset += 1
    
    # Align to 8 bytes for uint64_t
    offset = (offset + 7) & ~7
    
    # uint64_t total_requests_submitted
    total_submitted = struct.unpack_from('Q', shm.buf, offset)[0]
    offset += 8
    
    # uint64_t total_requests_completed
    total_completed = struct.unpack_from('Q', shm.buf, offset)[0]
    offset += 8
    
    # uint64_t total_batches_processed
    total_batches = struct.unpack_from('Q', shm.buf, offset)[0]
    offset += 8
    
    # uint32_t next_slot_hint
    next_hint = struct.unpack_from('I', shm.buf, offset)[0]
    offset += 4
    
    # Print server status
    print("\n📊 SERVER STATUS:")
    print(f"  Server Ready:        {'YES ✓' if server_ready else 'NO ✗'}")
    print(f"  Shutdown Requested:  {'YES' if shutdown_requested else 'NO'}")
    
    print("\n📈 STATISTICS:")
    print(f"  Total Submitted:     {total_submitted}")
    print(f"  Total Completed:     {total_completed}")
    print(f"  Total Batches:       {total_batches}")
    print(f"  Next Slot Hint:      {next_hint}")
    
    # Align to 64 bytes for request array
    offset = 64
    
    # Calculate structure sizes
    # EvalRequest: job_id (8) + state (1) + padding to 8 + board_state + legal_mask + padding to 64
    request_size = 64 + (INPUT_SIZE + POLICY_SIZE) * 4  # Align to cache line
    request_size = ((request_size + 63) // 64) * 64
    
    # EvalResponse: job_id (8) + policy + value + padding
    response_size = 64 + POLICY_SIZE * 4 + 4  # Align to cache line
    response_size = ((response_size + 63) // 64) * 64
    
    print(f"\n📐 STRUCTURE SIZES:")
    print(f"  Request size:        {request_size} bytes")
    print(f"  Response size:       {response_size} bytes")
    print(f"  Request array start: offset {offset}")
    print(f"  Response array start: offset {offset + request_size * MAX_BATCH_SIZE}")
    
    # Parse requests array
    state_counts = {i: 0 for i in range(5)}
    active_slots = []
    
    for slot in range(MAX_BATCH_SIZE):
        req_offset = offset + slot * request_size
        
        # Read job_id (uint64_t)
        job_id = struct.unpack_from('Q', shm.buf, req_offset)[0]
        
        # Read state (uint8_t at offset 8)
        state = struct.unpack_from('B', shm.buf, req_offset + 8)[0]
        
        state_counts[state] = state_counts.get(state, 0) + 1
        
        if state != JobState.FREE:
            # Read first few board values
            board_offset = req_offset + 16  # After job_id + state + padding
            board_values = struct.unpack_from('fff', shm.buf, board_offset)
            
            active_slots.append({
                'slot': slot,
                'job_id': job_id,
                'state': state,
                'board_sample': board_values
            })
    
    print(f"\n🔢 SLOT STATE SUMMARY:")
    print(f"  FREE:                {state_counts.get(JobState.FREE, 0)}")
    print(f"  WRITING:             {state_counts.get(JobState.WRITING, 0)}")
    print(f"  READY:               {state_counts.get(JobState.READY, 0)}")
    print(f"  PROCESSING:          {state_counts.get(JobState.PROCESSING, 0)}")
    print(f"  DONE:                {state_counts.get(JobState.DONE, 0)}")
    print(f"  TOTAL ACTIVE:        {len(active_slots)}")
    
    if active_slots:
        print(f"\n🔍 ACTIVE SLOT DETAILS:")
        for slot_info in active_slots:
            state_str = JobState.to_string(slot_info['state'])
            print(f"  Slot {slot_info['slot']:3d}: {state_str:10s} "
                  f"(job_id={slot_info['job_id']}) "
                  f"board[0:3]={slot_info['board_sample']}")
    
    # Check for stuck jobs
    print(f"\n⏰ CHECKING FOR STUCK JOBS...")
    time.sleep(0.1)
    
    stuck_jobs = []
    for slot_info in active_slots:
        if slot_info['state'] in (JobState.READY, JobState.PROCESSING):
            # Re-read state
            req_offset = offset + slot_info['slot'] * request_size
            state2 = struct.unpack_from('B', shm.buf, req_offset + 8)[0]
            
            if state2 == slot_info['state']:
                stuck_jobs.append(slot_info)
    
    if stuck_jobs:
        for slot_info in stuck_jobs:
            state_str = JobState.to_string(slot_info['state'])
            print(f"  ⚠️  STUCK: Slot {slot_info['slot']} in {state_str} state "
                  f"(job_id={slot_info['job_id']})")
    else:
        print(f"  ✓ No stuck jobs detected")
    
    print("\n" + "=" * 60)
    
    # Cleanup
    shm.close()

if __name__ == "__main__":
    import sys
    shm_name = "mcts_jax_inference"
    if len(sys.argv) > 1:
        shm_name = sys.argv[1]
    
    diagnose_shared_memory(shm_name)