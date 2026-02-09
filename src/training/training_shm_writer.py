"""
Training Shared Memory Reader - Python Side
============================================
Attaches to the /az_training segment created by C++.
Provides random batch sampling for JAX training.
"""

import time
import ctypes
import numpy as np
from typing import Dict, Optional, Tuple
from multiprocessing import shared_memory
import sys

from src.training.training_shm_protocol import (
    TrainingBufferHeader,
    TrainingPosition,
    verify_sizes,
    create_numpy_views,
    TRAINING_POSITION_BYTES,
)

class TrainingShmReader:
    """
    Reads training data from the /az_training shared memory segment.
    Handles startup synchronization and provides batch sampling.
    """
    
    def __init__(self, segment_name: str = "/az_training", 
                 max_retries: int = 60,
                 retry_interval: float = 1.0):
        """
        Attach to the training shared memory segment.
        
        Args:
            segment_name: SHM segment name (must match C++)
            max_retries: How many times to retry if segment doesn't exist
            retry_interval: Seconds to wait between retries
        """
        self.segment_name = segment_name
        self.shm = None
        self.shm_buffer = None
        self.header = None
        self.max_capacity = None
        self.views = None
        
        print(f"\n[TrainingShmReader] Connecting to segment: {segment_name}")
        
        # Verify protocol before attempting connection
        if not verify_sizes():
            raise RuntimeError("Protocol verification failed! Check struct layouts.")
        
        # Try to attach to shared memory (with retries for startup race)
        for attempt in range(max_retries):
            try:
                # Try to open existing segment (no create flag)
                self.shm = shared_memory.SharedMemory(
                    name=segment_name.lstrip('/'),
                    create=False
                )
                print(f"[TrainingShmReader] ✅ Connected to segment (size: {len(self.shm.buf) / 1024 / 1024:.1f} MB)")
                break
                
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    print(f"[TrainingShmReader] Segment not found, waiting... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_interval)
                else:
                    raise RuntimeError(
                        f"Failed to attach to {segment_name} after {max_retries} attempts. "
                        f"Is the C++ self-play process running?"
                    )
        
        # Map the buffer
        self.shm_buffer = memoryview(self.shm.buf)
        
        # Parse header
        self.header = TrainingBufferHeader.from_buffer(self.shm.buf)
        
        # Calculate max_capacity from segment size
        header_size = ctypes.sizeof(TrainingBufferHeader)
        positions_size = len(self.shm.buf) - header_size
        self.max_capacity = positions_size // TRAINING_POSITION_BYTES
        
        print(f"[TrainingShmReader] Buffer capacity: {self.max_capacity:,} positions")
        
        # Create NumPy views into shared memory
        print(f"[TrainingShmReader] Creating NumPy views...")
        self.views = create_numpy_views(self.shm_buffer, self.max_capacity)
        
        print(f"[TrainingShmReader] Views created:")
        for name, arr in self.views.items():
            print(f"  - {name:10} shape={arr.shape} dtype={arr.dtype}")
        
        print(f"[TrainingShmReader] ✅ Reader initialized\n")
    
    def wait_for_data(self, min_positions: int = 2000, 
                     print_interval: int = 5,
                     check_interval: float = 0.5):
        """
        Block until buffer contains at least min_positions.
        This is the warm-up gate before training starts.
        
        Args:
            min_positions: Minimum positions required
            print_interval: How often to print progress (seconds)
            check_interval: How often to check (seconds)
        """
        print(f"[TrainingShmReader] Waiting for {min_positions:,} positions...")
        
        last_print = time.time()
        
        while True:
            current_size = self.current_size
            
            if current_size >= min_positions:
                print(f"[TrainingShmReader] ✅ Buffer ready: {current_size:,}/{min_positions:,} positions\n")
                break
            
            # Print progress periodically
            if time.time() - last_print >= print_interval:
                print(f"[TrainingShmReader] Progress: {current_size:,}/{min_positions:,} positions "
                      f"({100.0 * current_size / min_positions:.1f}%)")
                last_print = time.time()
            
            time.sleep(check_interval)
    
    def sample_batch(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """
        Sample a random batch from the buffer.
        
        Args:
            batch_size: Number of positions to sample
            rng: NumPy random generator (creates one if None)
        
        Returns:
            Dictionary with keys: 'boards', 'pi', 'z', 'mask'
            All arrays are copies (safe to move to GPU)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Read current size atomically
        size = self.current_size
        
        if size == 0:
            raise RuntimeError("Buffer is empty! Cannot sample.")
        
        # Sample random indices in [0, size)
        indices = rng.integers(0, size, size=batch_size)
        
        # Gather data (this creates copies, not views)
        batch = {
            'boards': self.views['boards'][indices].copy(),  # [batch_size, 3, 4, 4]
            'pi': self.views['pi'][indices].copy(),          # [batch_size, 16]
            'z': self.views['z'][indices].copy(),            # [batch_size]
            'mask': self.views['mask'][indices].copy(),      # [batch_size, 16]
        }
        
        return batch
    
    @property
    def generation(self) -> int:
        """Current generation counter (bumped after each game flush)."""
        return self.header.generation
    
    @property
    def current_size(self) -> int:
        """Number of valid positions in buffer."""
        return self.header.current_size
    
    @property
    def write_index(self) -> int:
        """Next write position (wraps at max_capacity)."""
        return self.header.write_index
    
    def is_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self.header.shutdown
    
    def close(self):
        """Close the shared memory connection."""
        if self.shm is not None:
            self.shm.close()
            print(f"[TrainingShmReader] Disconnected from {self.segment_name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_stats(self) -> Dict[str, any]:
        """Get current buffer statistics."""
        return {
            'generation': self.generation,
            'current_size': self.current_size,
            'write_index': self.write_index,
            'max_capacity': self.max_capacity,
            'utilization': 100.0 * self.current_size / self.max_capacity,
            'is_shutdown': self.is_shutdown(),
        }


def main():
    """Test the reader by connecting and monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test training SHM reader')
    parser.add_argument('--segment', type=str, default='/az_training',
                       help='Shared memory segment name')
    parser.add_argument('--min-positions', type=int, default=100,
                       help='Wait for this many positions before starting')
    args = parser.parse_args()
    
    with TrainingShmReader(segment_name=args.segment) as reader:
        # Wait for initial data
        reader.wait_for_data(min_positions=args.min_positions)
        
        # Monitor for a bit
        print("\n[Monitor Mode] Press Ctrl+C to exit\n")
        last_gen = reader.generation
        
        try:
            while not reader.is_shutdown():
                current_gen = reader.generation
                
                if current_gen != last_gen:
                    stats = reader.get_stats()
                    print(f"Gen {stats['generation']:5d} | "
                          f"Size: {stats['current_size']:6,}/{stats['max_capacity']:,} "
                          f"({stats['utilization']:5.1f}%) | "
                          f"Write: {stats['write_index']:6,}")
                    
                    # Sample a small batch to test
                    batch = reader.sample_batch(batch_size=4)
                    print(f"  Sample batch: boards={batch['boards'].shape}, "
                          f"z_mean={batch['z'].mean():.3f}")
                    
                    last_gen = current_gen
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n[Monitor] Stopped")
        
        if reader.is_shutdown():
            print("\n[Monitor] Shutdown signal received")


if __name__ == "__main__":
    main()