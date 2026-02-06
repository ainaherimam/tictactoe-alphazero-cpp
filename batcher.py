"""
Inference Batcher

Collect requests into batches for efficient GPU inference.

Batching Strategy:
- Collect up to max_batch_size requests
- Wait up to max_wait_ms if we have some but not full batch
- Return immediately if full batch or no requests after extended wait
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List

from shared_memory_interface import SharedMemoryInterface
from shared_memory_protocol import INPUT_SIZE, POLICY_SIZE, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH


@dataclass
class Batch:
    """A batch of inference requests"""
    slots: List[int]                    # Slot indices
    job_ids: List[int]                  # Job IDs
    board_states: np.ndarray            # [batch, INPUT_SIZE]
    legal_masks: np.ndarray             # [batch, POLICY_SIZE]
    
    
    def __len__(self):
        return len(self.slots)
    
    def is_empty(self):
        return len(self.slots) == 0
    
    def reshape_for_model(self):
        """
        Reshape inputs for model consumption.
        
        Returns:
            board_states: [batch, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH]
            legal_masks: [batch, POLICY_SIZE]
        """
        batch_size = len(self.slots)
        
        # Reshape board states from [B, INPUT_SIZE] to [B, C, H, W]
        board_states = self.board_states.reshape(
            batch_size, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH
        )
        
        # Legal masks stay as [B, POLICY_SIZE]
        legal_masks = self.legal_masks
        
        return board_states, legal_masks


class InferenceBatcher:
    """
    Batching policy for inference requests.
    
    Adaptive batching:
    - Collects up to max_batch_size requests
    - If some requests available, waits up to max_wait_ms for more
    - If no requests, waits longer before giving up
    - Returns immediately when batch is full
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        min_batch_size: int = 8,      # ADD THIS
        max_wait_ms: float = 0.5,
        empty_wait_ms: float = 1.0
    ):
        """
        Args:
            max_batch_size: Maximum batch size
            max_wait_ms: Max time to wait for more requests (milliseconds)
            empty_wait_ms: Time to wait when completely empty (milliseconds)
        """
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.empty_wait_ms = empty_wait_ms / 1000.0
        
        # Statistics
        self.total_batches_collected = 0
        self.total_requests_collected = 0
        self.empty_polls = 0
    
    def collect_batch(self, shm: SharedMemoryInterface) -> Batch:
        """
        Collect a batch of requests from shared memory.
        
        KataGo-style batching:
        - Process immediately if batch >= MIN_BATCH_SIZE
        - Or if waited MAX_WAIT_MS
        - Brief sleep between scans
        
        Args:
            shm: Shared memory interface
            
        Returns:
            Batch object (may be empty if no requests available)
        """
        # Configuration
        slots = []
        job_ids = []
        board_states = []
        legal_masks = []
        
        start_time = time.time()
        
        while True:
            # Scan for READY requests
            ready_slots = shm.scan_ready_requests()
            
            # Try to claim each ready request
            for slot in ready_slots:
                if len(slots) >= self.max_batch_size:
                    break
                
                result = shm.claim_request(slot)
                if result is not None:
                    job_id, board_state, legal_mask = result
                    slots.append(slot)
                    job_ids.append(job_id)
                    board_states.append(board_state)
                    legal_masks.append(legal_mask)
            
            # Exit condition 1: Batch is full
            if len(slots) >= self.max_batch_size:
                break
            
            # Exit condition 2: Have minimum batch size
            if len(slots) >= self.min_batch_size:
                break
            
            # Exit condition 3: Timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= self.max_wait_ms:
                if len(slots) > 0:
                    # Have some requests - process them
                    break
                else:
                    # No requests after timeout - return empty
                    self.empty_polls += 1
                    return Batch([], [], np.array([]), np.array([]))
            
            # Brief sleep before next scan
            if len(slots) > 0:
                time.sleep(0.0005)  # 500μs - waiting for more
            else:
                time.sleep(0.001)   # 1ms - idle
        
        # Update statistics
        self.total_batches_collected += 1
        self.total_requests_collected += len(slots)
        
        # Stack arrays and create batch
        batch = Batch(
            slots=slots,
            job_ids=job_ids,
            board_states=np.stack(board_states, axis=0).astype(np.float32),
            legal_masks=np.stack(legal_masks, axis=0).astype(np.float32)
        )
        
        return batch
    
    def get_stats(self) -> dict:
        """Get batcher statistics"""
        avg_batch_size = 0.0
        if self.total_batches_collected > 0:
            avg_batch_size = self.total_requests_collected / self.total_batches_collected
        
        return {
            'total_batches': self.total_batches_collected,
            'total_requests': self.total_requests_collected,
            'avg_batch_size': avg_batch_size,
            'empty_polls': self.empty_polls
        }
    
    def print_stats(self):
        """Print batcher statistics"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("Batcher Statistics")
        print("=" * 60)
        print(f"Total batches:      {stats['total_batches']}")
        print(f"Total requests:     {stats['total_requests']}")
        print(f"Avg batch size:     {stats['avg_batch_size']:.2f}")
        print(f"Empty polls:        {stats['empty_polls']}")
        print("=" * 60 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing batcher.py\n")
    
    from shared_memory_interface import SharedMemoryInterface
    from shared_memory_protocol import JobState
    
    # Create shared memory
    print("Creating shared memory...")
    shm = SharedMemoryInterface("test_batcher_shm", create=True)
    
    # Create batcher
    print("Creating batcher...")
    batcher = InferenceBatcher(
        max_batch_size=8,
        max_wait_ms=10.0
    )
    
    # Test 1: Empty batch
    print("\nTest 1: Collecting from empty buffer")
    batch = batcher.collect_batch(shm)
    assert batch.is_empty()
    print(f"✓ Got empty batch: {len(batch)} requests")
    
    # Test 2: Add some requests manually
    print("\nTest 2: Adding mock requests")
    for i in range(5):
        req = shm.buffer.requests[i]
        req.job_id.value = i + 1
        req.state.value = JobState.READY
        
        # Fill with dummy data
        for j in range(INPUT_SIZE):
            req.board_state[j] = float(i * 0.1)
        for j in range(POLICY_SIZE):
            req.legal_mask[j] = 1.0
    
    print("Added 5 mock requests")
    
    # Collect batch
    print("Collecting batch...")
    batch = batcher.collect_batch(shm)
    print(f"✓ Got batch with {len(batch)} requests")
    
    assert len(batch) == 5
    assert batch.board_states.shape == (5, INPUT_SIZE)
    assert batch.legal_masks.shape == (5, POLICY_SIZE)
    
    # Test reshape
    print("\nTest 3: Reshaping for model")
    boards, masks = batch.reshape_for_model()
    print(f"  Board shape: {boards.shape}")
    print(f"  Mask shape:  {masks.shape}")
    assert boards.shape == (5, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    assert masks.shape == (5, POLICY_SIZE)
    print("✓ Reshape works correctly")
    
    # Print stats
    print("\nTest 4: Statistics")
    batcher.print_stats()
    
    # Cleanup
    shm.cleanup()
    
    print("✅ All tests passed!")