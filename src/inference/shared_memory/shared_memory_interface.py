"""
Shared Memory Interface (Python side)

Low-level interface to shared memory.
No ML logic here - just IPC primitives.

Responsibilities:
- Create/attach to shared memory
- Provide safe access to buffer
- Atomic state transitions
- No batching logic (that's in batcher.py)
- No ML logic (that's in inference_server.py)
"""

import ctypes
from typing import Optional, List, Tuple
import numpy as np
from multiprocessing import shared_memory

from src.inference.shared_memory.shared_memory_protocol import (
    SharedMemoryBuffer,
    EvalRequest,
    EvalResponse,
    JobState,
    MAX_BATCH_SIZE,
    INPUT_SIZE,
    POLICY_SIZE,
    initialize_buffer
)


class SharedMemoryInterface:
    """
    Python interface to shared memory buffer.
    
    This class handles the low-level IPC with C++ workers.
    """
    
    def __init__(self, shm_name: str, create: bool = True):
        """
        Initialize shared memory interface.
        
        Args:
            shm_name: Name of shared memory segment (e.g., "mcts_jax_inference")
            create: If True, create new shm; if False, attach to existing
        """
        # Remove leading '/' if present (multiprocessing.shared_memory handles naming)
        self.shm_name = shm_name.lstrip('/')
        self.buffer_size = ctypes.sizeof(SharedMemoryBuffer)
        self.shm = None
        self.buffer = None
        
        if create:
            self._create_shm()
        else:
            self._attach_shm()
    
    def _create_shm(self):
        """Create new shared memory region"""
        try:
            # Try to create new shared memory
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.buffer_size
            )
            print(f"[SharedMemory] ✓ Created: {self.shm_name} ({self.buffer_size} bytes)")
            
        except FileExistsError:
            # Clean up stale shared memory and retry
            print(f"[SHM] Removing stale shared memory: {self.shm_name}")
            try:
                stale_shm = shared_memory.SharedMemory(name=self.shm_name)
                stale_shm.close()
                stale_shm.unlink()
            except Exception as e:
                print(f"[SHM] Warning: Could not clean up stale memory: {e}")
            
            # Try creating again
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.buffer_size
            )
            print(f"[SharedMemory] ✓ Created: {self.shm_name} ({self.buffer_size} bytes)")
        
        # Create ctypes view of the shared memory buffer
        self.buffer = SharedMemoryBuffer.from_buffer(self.shm.buf)
        
        # Initialize to clean state
        initialize_buffer(self.buffer)
    
    def _attach_shm(self):
        """Attach to existing shared memory segment"""
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        
        # Create ctypes view of the shared memory buffer
        self.buffer = SharedMemoryBuffer.from_buffer(self.shm.buf)
        
        print(f"[SharedMemory] ✓ Attached: {self.shm_name}")
    
    # ========================================================================
    # SERVER CONTROL
    # ========================================================================
    
    def set_server_ready(self, ready: bool):
        """Signal that server is ready"""
        self.buffer.server_ready.value = ready
        if ready:
            print("[SharedMemory] Server marked as READY")
        else:
            print("[SharedMemory] Server marked as NOT READY")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested by C++"""
        return self.buffer.shutdown_requested.value
    
    # ========================================================================
    # REQUEST SCANNING
    # ========================================================================
    
    def scan_ready_requests(self) -> List[int]:
        """
        Find all requests in READY state.
        
        Returns:
            List of slot indices with READY requests
        """
        ready_slots = []
        for slot in range(MAX_BATCH_SIZE):
            state = self.buffer.requests[slot].state.value
            if state == JobState.READY:
                ready_slots.append(slot)
        return ready_slots
    
    def claim_request(self, slot: int) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Atomically claim a READY request (READY -> PROCESSING).
        
        Args:
            slot: Slot index to claim
            
        Returns:
            (job_id, board_state, legal_mask) if successful, None if failed
            
        Note: Python doesn't have true atomic CAS, but since only Python
              server reads READY and writes PROCESSING, this is safe.
        """
        req = self.buffer.requests[slot]
        
        # Check state
        if req.state.value != JobState.READY:
            return None
        
        # Read data while in READY state
        job_id = req.job_id.value
        
        # Copy input arrays (important: copy, don't reference!)
        board_state = np.ctypeslib.as_array(req.board_state).copy()
        legal_mask = np.ctypeslib.as_array(req.legal_mask).copy()
        
        # Mark as PROCESSING
        # This releases the C++ thread from spin-waiting
        req.state.value = JobState.PROCESSING
        
        return (job_id, board_state, legal_mask)
    
    # ========================================================================
    # RESPONSE WRITING
    # ========================================================================
    
    def write_response(
        self,
        slot: int,
        job_id: int,
        policy: np.ndarray,
        value: float
    ):
        """
        Write response and mark DONE.
        
        Args:
            slot: Slot index
            job_id: Job ID (for verification)
            policy: Policy array [POLICY_SIZE]
            value: Value scalar
        """
        req = self.buffer.requests[slot]
        resp = self.buffer.responses[slot]
        
        # Verify we're processing this job
        current_state = req.state.value
        if current_state != JobState.PROCESSING:
            raise RuntimeError(
                f"Cannot write response: slot {slot} is in state "
                f"{JobState(current_state).name}, expected PROCESSING")
        
        current_job_id = req.job_id.value
        if current_job_id != job_id:
            raise RuntimeError(f"Job ID in slot {slot}: expected {job_id}, got {current_job_id}")
        
        # Verify policy shape
        if policy.shape != (POLICY_SIZE,):
            raise ValueError(
                f"Policy shape mismatch: expected ({POLICY_SIZE},), "
                f"got {policy.shape}")
        
        # Write response data
        resp.job_id.value = job_id
        
        # Copy policy
        for i in range(POLICY_SIZE):
            resp.policy[i] = float(policy[i])
        
        resp.value = float(value)
        
        # Memory barrier (ensure all writes visible)
        # In CPython, this is implicit due to GIL, but good practice
        ctypes.pythonapi.PyEval_InitThreads()
        
        # Mark DONE (this releases the C++ thread!)
        req.state.value = JobState.DONE
        
        # Update stats
        self.buffer.total_requests_completed.value += 1
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'submitted': self.buffer.total_requests_submitted.value,
            'completed': self.buffer.total_requests_completed.value,
            'batches': self.buffer.total_batches_processed.value,
            'pending': self._count_pending()
        }
    
    def _count_pending(self) -> int:
        """Count non-FREE slots"""
        count = 0
        for i in range(MAX_BATCH_SIZE):
            state = self.buffer.requests[i].state.value
            if state != JobState.FREE:
                count += 1
        return count
    
    def increment_batch_count(self):
        """Increment total batches processed"""
        self.buffer.total_batches_processed.value += 1

    def notify_batch_complete(self):
        """Increment notification counter to wake C++ waiters"""
        current = self.buffer.notification_counter.value
        self.buffer.notification_counter.value = current + 1
    
    # ========================================================================
    # DEBUGGING
    # ========================================================================
    
    def print_buffer_state(self):
        """Print current buffer state (for debugging)"""
        print("\n" + "=" * 60)
        print("Buffer State")
        print("=" * 60)
        print(f"Server ready:     {self.buffer.server_ready.value}")
        print(f"Shutdown:         {self.buffer.shutdown_requested.value}")
        print(f"Submitted:        {self.buffer.total_requests_submitted.value}")
        print(f"Completed:        {self.buffer.total_requests_completed.value}")
        print(f"Batches:          {self.buffer.total_batches_processed.value}")
        print(f"Pending:          {self._count_pending()}")
        
        # Count by state
        state_counts = {state: 0 for state in JobState}
        for i in range(MAX_BATCH_SIZE):
            state = JobState(self.buffer.requests[i].state.value)
            state_counts[state] += 1
        
        print("\nSlot states:")
        for state, count in state_counts.items():
            if count > 0:
                print(f"  {state.name:12} : {count:3}")
        print("=" * 60 + "\n")
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def cleanup(self, unlink: bool = False):
        """
        Cleanup resources
        
        Args:
            unlink: If True, remove the shared memory segment (only do this
                   when the last process is done with it)
        """
        print("[SharedMemory] Cleaning up...")
        
        if self.shm:
            try:
                self.shm.close()
                if unlink:
                    self.shm.unlink()
                    print(f"[SharedMemory] Removed: {self.shm_name}")
            except Exception as e:
                print(f"[SharedMemory] Cleanup warning: {e}")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.cleanup()
        return False


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing shared_memory_interface.py\n")
    
    # Test creating interface
    print("Test 1: Create shared memory interface")
    shm = SharedMemoryInterface("test_shm", create=True)
    print("✓ Interface created\n")
    
    # Test server ready
    print("Test 2: Server ready flag")
    shm.set_server_ready(True)
    assert shm.buffer.server_ready.value == True
    print("✓ Server ready flag works\n")
    
    # Test statistics
    print("Test 3: Statistics")
    stats = shm.get_stats()
    print(f"  Stats: {stats}")
    print("✓ Statistics work\n")
    
    # Test buffer state
    print("Test 4: Buffer state")
    shm.print_buffer_state()
    print("✓ Buffer state printing works\n")
    
    # Cleanup (with unlink since this is the only process)
    shm.cleanup(unlink=True)
    
    print("✅ All tests passed!")