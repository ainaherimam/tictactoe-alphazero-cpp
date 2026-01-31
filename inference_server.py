#!/usr/bin/env python3
"""
Inference Server

Main Python process that:
1. Creates shared memory
2. Loads JAX model
3. Collects batches
4. Runs inference
5. Writes responses

This is the ONLY Python process that runs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
import signal
import sys
from pathlib import Path

from shared_memory_interface import SharedMemoryInterface
from batcher import InferenceBatcher
from shared_memory_protocol import (
    INPUT_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    POLICY_SIZE,
    verify_sizes
)

# Import your model
from alphazero_model import AlphaZeroModel


class InferenceServer:
    """
    Main inference server.
    
    Owns:
    - Shared memory
    - JAX model
    - Batching logic
    - Main loop
    """
    
    def __init__(
        self,
        model: AlphaZeroModel,
        shm_name: str = "mcts_jax_inference",
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
        log_interval: int = 100
    ):
        """
        Args:
            model: Loaded AlphaZeroModel
            shm_name: Shared memory name (without leading /)
            max_batch_size: Maximum batch size
            max_wait_ms: Max wait time for batching (ms)
            log_interval: Log stats every N batches
        """
        self.model = model
        self.log_interval = log_interval
        
        # Create shared memory
        print("\n" + "=" * 60)
        print("Creating Shared Memory")
        print("=" * 60)
        self.shm = SharedMemoryInterface(shm_name, create=True)
        
        # Create batcher
        print("\n" + "=" * 60)
        print("Creating Batcher")
        print("=" * 60)
        self.batcher = InferenceBatcher(
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms
        )
        print(f"Max batch size: {max_batch_size}")
        print(f"Max wait:       {max_wait_ms}ms")
        
        # Statistics
        self.total_batches = 0
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        
        # Shutdown flag
        self.should_shutdown = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n[Server] Received shutdown signal...")
        self.should_shutdown = True
    
    def run(self):
        """Main server loop"""
        try:
            # Signal ready
            self.shm.set_server_ready(True)
            print("\n" + "=" * 60)
            print("🚀 INFERENCE SERVER READY")
            print("=" * 60)
            print("Waiting for C++ workers to submit requests...")
            print("Press Ctrl+C to shutdown gracefully")
            print("=" * 60 + "\n")
            
            idle_count = 0
            max_idle = 5000  # Print idle message every 100 empty polls
            
            while not self.should_shutdown and not self.shm.is_shutdown_requested():

                batch = self.batcher.collect_batch(self.shm)
                if batch.is_empty():
                    idle_count += 1
                    if idle_count >= max_idle:
                        print("[Server] Idle - waiting for requests...")
                        idle_count = 0
                    time.sleep(0.001)  # 1ms idle sleep
                    continue
                
                idle_count = 0  # Reset idle counter
                
                # Process batch
                self._process_batch(batch)

                
        except KeyboardInterrupt:
            print("\n[Server] Interrupted...")
        except Exception as e:
            print(f"\n[Server] ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._shutdown()
    
    def _process_batch(self, batch):
        """Process a batch of requests"""
        batch_size = len(batch)
        
        # Reshape for model
        board_states, legal_masks = batch.reshape_for_model()
        
        # Convert to JAX arrays
        inputs = jnp.array(board_states, dtype=jnp.float32)
        masks = jnp.array(legal_masks, dtype=jnp.float32)
        
        # Run inference
        start = time.time()
        policy_probs, values = self._run_inference(inputs, masks)
        inference_time = time.time() - start
        
        # Convert to numpy for writing back
        policy_probs = np.array(policy_probs)
        values = np.array(values)
        
        # Write responses
        for i in range(batch_size):
            try:
                self.shm.write_response(
                    slot=batch.slots[i],
                    job_id=batch.job_ids[i],
                    policy=policy_probs[i],
                    value=float(values[i])
                )
            except Exception as e:
                print(f"[Server] ERROR writing response for slot {batch.slots[i]}: {e}")
        
        # Update stats
        self.total_batches += 1
        self.total_requests += batch_size
        self.total_inference_time += inference_time
        self.shm.increment_batch_count()
        
        # Log periodically
        if self.total_batches % self.log_interval == 0:
            self._log_stats(batch_size, inference_time)
    
    def _run_inference(self, inputs, masks):
        """
        Run JAX inference using JIT-compiled prediction.
        Args:
            inputs: [B, C, H, W] board states
            masks: [B, POLICY_SIZE] legal move masks
        Returns:
            policy: [B, POLICY_SIZE] (probabilities)
            values: [B] (value estimates)
        """
        # Use the JIT-compiled predict method
        policy_logits, values = self.model.predict(inputs)
        
        # Apply legal masks (set illegal moves to -inf)
        policy_logits = jnp.where(masks > 0, policy_logits, -1e9)
        
        # Softmax to get probabilities
        policy = jax.nn.softmax(policy_logits, axis=-1)
        
        # Zero out illegal moves (cleaner than -inf)
        policy = jnp.where(masks > 0.5, policy, 0.0)
    
        return policy, values
    
    def _log_stats(self, last_batch_size, last_inference_time):
        """Log statistics"""
        elapsed = time.time() - self.start_time
        avg_batch_size = self.total_requests / self.total_batches if self.total_batches > 0 else 0
        avg_inference_time = self.total_inference_time / self.total_batches if self.total_batches > 0 else 0
        throughput = self.total_requests / elapsed if elapsed > 0 else 0
        
        print(f"[Server] Batch {self.total_batches:6d} | "
              f"size={last_batch_size:3d} | "
              f"time={last_inference_time*1000:6.2f}ms | "
              f"avg_size={avg_batch_size:5.2f} | "
              f"avg_time={avg_inference_time*1000:6.2f}ms | "
              f"throughput={throughput:7.1f} req/s")
    
    def _shutdown(self):
        """Graceful shutdown"""
        print("\n" + "=" * 60)
        print("Shutting Down")
        print("=" * 60)
        
        # Mark server as not ready
        self.shm.set_server_ready(False)
        
        # Print final statistics
        self._print_final_stats()
        
        # Cleanup
        self.shm.cleanup()
        
        print("\n" + "=" * 60)
        print("✓ Shutdown Complete")
        print("=" * 60 + "\n")
    
    def _print_final_stats(self):
        """Print final statistics"""
        elapsed = time.time() - self.start_time
        
        print("\nFinal Statistics:")
        print("-" * 60)
        print(f"Total batches:        {self.total_batches}")
        print(f"Total requests:       {self.total_requests}")
        print(f"Total time:           {elapsed:.1f}s")
        
        if self.total_batches > 0:
            avg_batch_size = self.total_requests / self.total_batches
            avg_inference_time = self.total_inference_time / self.total_batches
            print(f"Avg batch size:       {avg_batch_size:.2f}")
            print(f"Avg inference time:   {avg_inference_time*1000:.2f}ms")
        
        if elapsed > 0:
            throughput = self.total_requests / elapsed
            print(f"Throughput:           {throughput:.1f} req/s")
        
        # Batcher stats
        batcher_stats = self.batcher.get_stats()
        print(f"\nBatcher empty polls:  {batcher_stats['empty_polls']}")
        
        # Shared memory stats
        shm_stats = self.shm.get_stats()
        print(f"\nShared memory stats:")
        print(f"  Submitted:          {shm_stats['submitted']}")
        print(f"  Completed:          {shm_stats['completed']}")
        print(f"  Pending:            {shm_stats['pending']}")
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero JAX Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to model checkpoint')
    parser.add_argument('--shm-name', type=str, 
                       default='mcts_jax_inference',
                       help='Shared memory name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Maximum batch size')
    parser.add_argument('--max-wait-ms', type=float, default=5.0,
                       help='Max wait time for batching (ms)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Log stats every N batches')
    parser.add_argument('--num-channels', type=int, default=64,
                       help='Number of channels in model')
    parser.add_argument('--num-res-blocks', type=int, default=3,
                       help='Number of residual blocks')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("AlphaZero JAX Inference Server")
    print("=" * 60)
    print(f"Checkpoint:      {args.checkpoint or 'Random initialization'}")
    print(f"Shared memory:   {args.shm_name}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Max wait:        {args.max_wait_ms}ms")
    print(f"Log interval:    {args.log_interval}")
    print("=" * 60)
    
    # Verify protocol sizes
    print("\nVerifying Protocol...")
    print("=" * 60)
    if not verify_sizes():
        print("\n❌ ERROR: Protocol size mismatch!")
        print("This will cause memory corruption with C++!")
        return 1
    
    # Load model
    print("\nLoading Model...")
    print("=" * 60)
    
    model = AlphaZeroModel(
        num_actions=POLICY_SIZE,
        input_channels=INPUT_CHANNELS,
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks
    )
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load(str(checkpoint_path))
        else:
            print(f"⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print("⚠️  Initializing random model")
            model.initialize()
    else:
        print("Initializing random model (no checkpoint provided)")
        model.initialize()
    
    # Start server
    print("\nStarting Server...")
    print("=" * 60)
    
    server = InferenceServer(
        model=model,
        shm_name=args.shm_name,
        max_batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        log_interval=args.log_interval
    )
    
    server.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())