#!/usr/bin/env python3
"""
Inference Server

Main Python process that:
1. Creates shared memory
2. Loads JAX model (inference-only, no optimizer)
3. Collects batches
4. Runs inference
5. Writes responses
6. Hot-reloads new checkpoints without downtime

This is the ONLY Python process that runs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from shared_memory_interface import SharedMemoryInterface
from batcher import InferenceBatcher
from shared_memory_protocol import (
    POLICY_SIZE,
    verify_sizes,
)

from alphazero_model import (
    AlphaZeroNet,
    create_inference_state,
    load_checkpoint,
    load_checkpoint_for_inference,
)


class InferenceServer:
    """
    Main inference server.
    
    Owns:
    - Shared memory
    - JAX model (inference state only - no optimizer)
    - Batching logic
    - Hot-reload capability
    - Main loop
    """
    
    def __init__(
        self,
        model: AlphaZeroNet,
        inference_state: Dict[str, Any],
        shm_name: str = "mcts_jax_inference",
        max_batch_size: int = 32,
        max_wait_ms: float = 1.0,
        log_interval: int = 10000,
        checkpoint_watch_dir: Optional[str] = None,
    ):
        """
        Args:
            model: AlphaZeroNet nn.Module (architecture only)
            inference_state: Dict with 'params', 'batch_stats', 'apply_fn'
            shm_name: Shared memory name (without leading /)
            max_batch_size: Maximum batch size
            max_wait_ms: Max wait time for batching (ms)
            log_interval: Log stats every N batches
            checkpoint_watch_dir: Directory to watch for checkpoint reload triggers (None to disable)
        """
        self.model = model
        self.inference_state = inference_state
        self.log_interval = log_interval
        self.checkpoint_watch_dir = Path(checkpoint_watch_dir) if checkpoint_watch_dir else None
        self.shm_name = shm_name  # Store for reload sentinel matching
        
        # Track latest checkpoint
        self.latest_checkpoint_step = 0
        
        # Lock for atomic state swapping
        self.state_lock = threading.Lock()
        
        # Background reload thread
        self.reload_thread = None
        self.reload_in_progress = False
        
        # JIT-compile inference once. Closure captures model.apply (a pure
        # function); all runtime data (params, batch_stats, inputs, masks)
        # are passed as traced array arguments so JAX never has to retrace.
        @jax.jit
        def _inference(params, batch_stats, inputs, masks):
            log_policy, values = model.apply(
                {'params': params, 'batch_stats': batch_stats},
                inputs,
                masks,
                training=False,
            )
            return jnp.exp(log_policy), values
        
        self._inference = _inference
        
        # Pre-warm JIT with current state
        print("\n" + "=" * 60)
        print("Pre-warming JIT compilation...")
        print("=" * 60)
        dummy_input = jnp.zeros((1, 3, 4, 4))
        dummy_mask = jnp.ones((1, POLICY_SIZE))
        _ = self._inference(
            self.inference_state['params'],
            self.inference_state['batch_stats'],
            dummy_input,
            dummy_mask
        )
        print("✓ JIT compilation complete")
        
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
        self.total_reloads = 0
        
        # Shutdown flag
        self.should_shutdown = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n[Server] Received shutdown signal...")
        self.should_shutdown = True
    
    def _start_checkpoint_watcher(self):
        """Start background thread to watch for checkpoint reload sentinel files"""
        if self.checkpoint_watch_dir is None:
            return
        
        def watch_sentinel_file():
            print(f"\n[Reload] 🔍 Watching for sentinel files: {self.checkpoint_watch_dir}/reload.*.*.trigger")
            print(f"[Reload] 🏷️  This instance SHM name: {self.shm_name}")
            
            while not self.should_shutdown:
                try:
                    # Look for any reload.{shm_name}.{number}.trigger files
                    sentinel_files = list(self.checkpoint_watch_dir.glob("reload.*.*.trigger"))
                    
                    for sentinel in sentinel_files:
                        # Parse sentinel filename: reload.{shm_name}.{number}.trigger
                        try:
                            # Extract parts from reload.SHM_NAME.NUMBER.trigger
                            parts = sentinel.stem.split('.')  # ['reload', 'SHM_NAME', 'NUMBER']
                            if len(parts) != 3 or parts[0] != 'reload':
                                print(f"[Reload] ⚠️  Invalid sentinel format: {sentinel.name} (expected reload.SHM_NAME.NUMBER.trigger)")
                                sentinel.unlink()
                                continue
                            
                            sentinel_shm_name = parts[1]
                            checkpoint_step = int(parts[2])
                            
                        except (ValueError, IndexError) as e:
                            print(f"[Reload] ⚠️  Failed to parse sentinel {sentinel.name}: {e}")
                            sentinel.unlink()
                            continue
                        
                        # Check if this sentinel is for this instance
                        if sentinel_shm_name != self.shm_name:
                            # Not for us, skip (but don't delete - it's for another instance)
                            continue
                        
                        # This sentinel is for us!
                        print(f"\n[Reload] 🔔 Reload trigger detected for {sentinel_shm_name}: {sentinel.name}")
                        
                        # Delete sentinel file immediately
                        sentinel.unlink()

                        time.sleep(2.0)
                        
                        # Build checkpoint path
                        checkpoint_path = self.checkpoint_watch_dir / f"checkpoint_{checkpoint_step}"
                        
                        if not checkpoint_path.exists():
                            print(f"[Reload] ❌ Checkpoint not found: {checkpoint_path}")
                            continue
                        
                        if checkpoint_step > self.latest_checkpoint_step:
                            print(f"[Reload] 🆕 Loading checkpoint: step {checkpoint_step}")
                            self._reload_checkpoint_background(checkpoint_path, checkpoint_step)
                        else:
                            print(f"[Reload] ⏭️  Skipping checkpoint {checkpoint_step} (current: {self.latest_checkpoint_step})")
                    
                except Exception as e:
                    print(f"[Reload] ⚠️  Error checking for sentinel files: {e}")
                
                # Check every second for sentinel files
                time.sleep(1.0)
        
        self.reload_thread = threading.Thread(target=watch_sentinel_file, daemon=True)
        self.reload_thread.start()
    
    def _find_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Find the latest checkpoint in watch directory"""
        if not self.checkpoint_watch_dir.exists():
            return None
        
        checkpoints = []
        for ckpt_dir in self.checkpoint_watch_dir.glob("checkpoint_*"):
            if ckpt_dir.is_dir():
                try:
                    step = int(ckpt_dir.name.split("_")[1])
                    checkpoints.append({'path': ckpt_dir, 'step': step})
                except (ValueError, IndexError):
                    continue
        
        if not checkpoints:
            return None
        
        # Return latest by step number
        return max(checkpoints, key=lambda x: x['step'])
    
    def _reload_checkpoint_background(self, checkpoint_path: Path, step: int):
        """
        Hot-reload checkpoint in background without blocking inference.
        Uses pre-JIT strategy for zero-downtime reloading.
        """
        if self.reload_in_progress:
            print(f"[Reload] ⏭️  Skipping reload (already in progress)")
            return
        
        self.reload_in_progress = True
        print(f"[Reload] 🔄 Starting background reload from step {step}")
        reload_start = time.time()
        
        try:
            # 1. Load new checkpoint (doesn't block inference)
            print(f"[Reload]   Loading checkpoint data...")
            new_state = load_checkpoint_for_inference(
                checkpoint_path=str(checkpoint_path),
                num_channels=self.model.num_channels,
                num_res_blocks=self.model.num_res_blocks,
                num_actions=POLICY_SIZE,
            )
            
            # 2. Pre-compile with dummy input (blocks here, but not main thread)
            print(f"[Reload]   Pre-compiling JIT...")
            dummy_input = jnp.zeros((1, 3, 4, 4))
            dummy_mask = jnp.ones((1, POLICY_SIZE))
            _ = self._inference(
                new_state['params'],
                new_state['batch_stats'],
                dummy_input,
                dummy_mask
            )
            
            # 3. Atomic swap (instant, no inference disruption)
            with self.state_lock:
                self.inference_state = new_state
                self.latest_checkpoint_step = step
            
            reload_time = time.time() - reload_start
            self.total_reloads += 1
            
            print(f"[Reload] ✅ Reload complete in {reload_time:.2f}s (step {step})")
            
        except Exception as e:
            print(f"[Reload] ❌ Reload failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.reload_in_progress = False
    
    def reload_checkpoint_now(self, checkpoint_path: str):
        """
        Manually trigger checkpoint reload (blocks caller, not main inference loop).
        Useful for explicit reload commands.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"[Reload] ❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            step = int(checkpoint_path.name.split("_")[1])
        except (ValueError, IndexError):
            print(f"[Reload] ❌ Invalid checkpoint name format: {checkpoint_path.name}")
            return False
        
        self._reload_checkpoint_background(checkpoint_path, step)
        return True
    
    def run(self):
        """Main server loop"""
        try:
            # Start checkpoint watcher if enabled
            self._start_checkpoint_watcher()
            
            # Signal ready
            self.shm.set_server_ready(True)
            print("\n" + "=" * 60)
            print("🚀 INFERENCE SERVER READY")
            print("=" * 60)
            print("Waiting for C++ workers to submit requests...")
            if self.checkpoint_watch_dir:
                print(f"Hot-reload enabled: watching {self.checkpoint_watch_dir}")
            print("Press Ctrl+C to shutdown gracefully")
            print("=" * 60 + "\n")
            
            idle_count = 0
            max_idle = 5000  # Print idle message every 5000 empty polls
            
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
        
        # Run inference (thread-safe read of current state)
        start = time.time()
        policy_probs, values = self._run_inference(inputs, masks)
        # Block until compute is actually done, then transfer to host.
        # Both np.array() calls are inside the timed window so the
        # logged latency includes real compute, not just dispatch.
        policy_probs = np.array(policy_probs)
        values = np.array(values)
        inference_time = time.time() - start
        
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
        
        self.shm.notify_batch_complete()
        
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
        Run JIT-compiled inference (thread-safe).
        Args:
            inputs: [B, C, H, W] board states
            masks: [B, POLICY_SIZE] legal move masks
        Returns:
            policy: [B, POLICY_SIZE] (probabilities, illegal moves are 0)
            values: [B] (value estimates)
        """
        # Thread-safe read of current state
        with self.state_lock:
            params = self.inference_state['params']
            batch_stats = self.inference_state['batch_stats']
        
        return self._inference(params, batch_stats, inputs, masks)
    
    def _log_stats(self, last_batch_size, last_inference_time):
        """Log statistics"""
        elapsed = time.time() - self.start_time
        avg_batch_size = self.total_requests / self.total_batches if self.total_batches > 0 else 0
        avg_inference_time = self.total_inference_time / self.total_batches if self.total_batches > 0 else 0
        throughput = self.total_requests / elapsed if elapsed > 0 else 0
        
        reload_info = f"| reloads={self.total_reloads}" if self.checkpoint_watch_dir else ""
        
        print(f"[Server] Batch {self.total_batches:6d} | "
              f"size={last_batch_size:3d} | "
              f"time={last_inference_time*1000:6.2f}ms | "
              f"avg_size={avg_batch_size:5.2f} | "
              f"avg_time={avg_inference_time*1000:6.2f}ms | "
              f"throughput={throughput:7.1f} req/s {reload_info}")
    
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
        print(f"Total reloads:        {self.total_reloads}")
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
    parser.add_argument('--log-interval', type=int, default=10000,
                       help='Log stats every N batches')
    parser.add_argument('--num-channels', type=int, default=64,
                       help='Number of channels in model')
    parser.add_argument('--num-res-blocks', type=int, default=3,
                       help='Number of residual blocks')
    parser.add_argument('--watch-checkpoints', type=str, default=None,
                       help='Directory to watch for checkpoint reload triggers (enables hot-reload)')
    
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
    if args.watch_checkpoints:
        print(f"Hot-reload:      {args.watch_checkpoints} (sentinel-triggered)")
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
    
    # Construct the bare nn.Module
    model = AlphaZeroNet(
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_actions=POLICY_SIZE,
    )
    
    # Initialize inference state (lightweight, no optimizer)
    rng = jax.random.PRNGKey(0)
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            inference_state = load_checkpoint_for_inference(
                checkpoint_path=str(checkpoint_path),
                num_channels=args.num_channels,
                num_res_blocks=args.num_res_blocks,
                num_actions=POLICY_SIZE,
            )
            # Extract step number for tracking
            try:
                initial_step = int(checkpoint_path.name.split("_")[1])
            except:
                initial_step = 0
        else:
            print(f"⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print("⚠️  Continuing with random weights")
            inference_state = create_inference_state(
                rng, args.num_channels, args.num_res_blocks, POLICY_SIZE
            )
            initial_step = 0
    else:
        print("Initializing random model (no checkpoint provided)")
        inference_state = create_inference_state(
            rng, args.num_channels, args.num_res_blocks, POLICY_SIZE
        )
        initial_step = 0
    
    # Start server
    print("\nStarting Server...")
    print("=" * 60)
    
    server = InferenceServer(
        model=model,
        inference_state=inference_state,
        shm_name=args.shm_name,
        max_batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        log_interval=args.log_interval,
        checkpoint_watch_dir=args.watch_checkpoints,
    )
    
    # Set initial checkpoint step for reload tracking
    server.latest_checkpoint_step = initial_step
    
    server.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())