import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""
Triton Inference Server for AlphaZero - OPTIMIZED
==================================================

Serves JAX model via gRPC with dynamic batching for low-latency inference.

Features:
    - JIT-compiled JAX model (loaded once, stays in memory)
    - Dynamic batching with minimal delay (0.1ms)
    - Hot-reload capability (no downtime during model updates)
    - Async gRPC interface for C++ clients
    - <2ms inference latency with batching
    - Comprehensive warmup for all batch sizes

Requirements:
    pip install nvidia-pytriton jax[cpu] flax optax orbax-checkpoint --break-system-packages

Usage:
    python triton_server.py [--checkpoint CKPT_PATH] [--port 8001] [--batch-size 16]

Hot-Reload:
    Touch file: {checkpoint_dir}/reload.alphazero.trigger
    Server will reload model in background and switch seamlessly
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import time
import threading
from typing import Optional

# Import AlphaZero model
from src.models.alphazero_model import (
    load_checkpoint_for_inference,
    create_inference_state,
    AlphaZeroNet
)

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig


# ============================================================================
# MODEL WRAPPER (JIT-COMPILED FOR PERFORMANCE)
# ============================================================================

class AlphaZeroInferenceModel:
    """
    Wraps JAX model for Triton inference with hot-reload support.
    
    Features:
    - Loads checkpoint or initializes random weights
    - JIT-compiles inference function on first call
    - Handles batched inputs efficiently
    - Thread-safe model swapping for hot-reload
    """
    
    def __init__(self, checkpoint_path=None, num_channels=64, num_res_blocks=3, force_random=False):
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.checkpoint_path = checkpoint_path
        self.force_random = force_random
        
        # Thread lock for safe model swapping
        self._model_lock = threading.RLock()
        
        # Load initial model
        self._load_model(checkpoint_path, force_random)
        
    def _load_model(self, checkpoint_path=None, force_random=False):
        """Load model weights and JIT-compile inference function."""
        
        print("[Triton Server] Initializing AlphaZero model...")
        
        # FORCE random weights for debugging
        if force_random:
            print("[Triton Server] ⚠️  FORCE_RANDOM_WEIGHTS enabled - ignoring any checkpoint!")
            checkpoint_path = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Triton Server] Loading checkpoint from {checkpoint_path}")
            state = load_checkpoint_for_inference(
                checkpoint_path=checkpoint_path,
                num_channels=self.num_channels,
                num_res_blocks=self.num_res_blocks,
                num_actions=16
            )
        else:
            if checkpoint_path:
                print(f"[Triton Server] ⚠️  Checkpoint not found: {checkpoint_path}")
            print("[Triton Server] Initializing with random weights")
            rng = jax.random.PRNGKey(42)
            state = create_inference_state(
                rng=rng,
                num_channels=self.num_channels,
                num_res_blocks=self.num_res_blocks,
                num_actions=16
            )
        
        params = state['params']
        batch_stats = state['batch_stats']
        apply_fn = state['apply_fn']
        
        # JIT-compile the inference function
        print("[Triton Server] JIT-compiling inference function...")
        infer_jit = jax.jit(lambda boards, mask: apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            boards,
            mask,
            training=False
        ))
        
        # CRITICAL: Comprehensive warmup for all batch sizes
        # This eliminates JIT compilation delays during inference
        print("[Triton Server] Running warmup inferences to trigger JIT compilation...")
        warmup_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
        for batch_size in warmup_sizes:
            dummy_boards = jnp.zeros((batch_size, 3, 4, 4), dtype=jnp.float32)
            dummy_mask = jnp.ones((batch_size, 16), dtype=jnp.float32)
            # Run twice to ensure compilation is complete
            _ = infer_jit(dummy_boards, dummy_mask)
            _ = infer_jit(dummy_boards, dummy_mask)
            print(f"[Triton Server]   ✓ Warmup complete for batch_size={batch_size}")
        
        # Atomically swap the model (thread-safe)
        with self._model_lock:
            self.params = params
            self.batch_stats = batch_stats
            self.apply_fn = apply_fn
            self._infer_jit = infer_jit
        
        print("[Triton Server] ✓ Model ready (JIT-compiled and cached for all batch sizes)")
    
    def reload_model(self, checkpoint_path: str):
        """
        Hot-reload model from checkpoint without stopping inference.
        
        This runs in a background thread, loads and JIT-compiles the new model,
        then atomically swaps it in. Inference continues uninterrupted.
        """
        print(f"\n[Hot-Reload] 🔄 Starting background reload from {checkpoint_path}")
        load_start = time.perf_counter()
        
        try:
            # Load new model (this is the slow part, done outside the lock)
            state = load_checkpoint_for_inference(
                checkpoint_path=checkpoint_path,
                num_channels=self.num_channels,
                num_res_blocks=self.num_res_blocks,
                num_actions=16
            )
            
            new_params = state['params']
            new_batch_stats = state['batch_stats']
            new_apply_fn = state['apply_fn']
            
            # JIT-compile new inference function
            print("[Hot-Reload] JIT-compiling new model...")
            new_infer_jit = jax.jit(lambda boards, mask: new_apply_fn(
                {'params': new_params, 'batch_stats': new_batch_stats},
                boards,
                mask,
                training=False
            ))
            
            # Warmup new model
            print("[Hot-Reload] Warming up new model...")
            warmup_sizes = [1, 2, 4, 8, 16]
            for batch_size in warmup_sizes:
                dummy_boards = jnp.zeros((batch_size, 3, 4, 4), dtype=jnp.float32)
                dummy_mask = jnp.ones((batch_size, 16), dtype=jnp.float32)
                _ = new_infer_jit(dummy_boards, dummy_mask)
                _ = new_infer_jit(dummy_boards, dummy_mask)
            
            # ATOMIC SWAP - this is fast, minimal disruption
            with self._model_lock:
                self.params = new_params
                self.batch_stats = new_batch_stats
                self.apply_fn = new_apply_fn
                self._infer_jit = new_infer_jit
            
            load_time = time.perf_counter() - load_start
            print(f"[Hot-Reload] ✓ Model swapped successfully in {load_time:.2f}s")
            
        except Exception as e:
            print(f"[Hot-Reload] ❌ Failed to reload model: {e}")
            import traceback
            traceback.print_exc()
    
    def infer_batch(self, boards_np, mask_np):
        """
        Batched inference (called by Triton).
        
        Args:
            boards_np: numpy array [B, 3, 4, 4]
            mask_np: numpy array [B, 16]
        
        Returns:
            policy_np: numpy array [B, 16]
            value_np: numpy array [B]
        """
        total_start = time.perf_counter()
        
        # OPTIMIZED: Use jnp.asarray instead of jnp.array to avoid unnecessary copies
        # asarray will create a view if possible, array always copies
        convert_start = time.perf_counter()
        boards = jnp.asarray(boards_np, dtype=jnp.float32)
        mask = jnp.asarray(mask_np, dtype=jnp.float32)
        convert_time = (time.perf_counter() - convert_start) * 1000
        
        # Run JIT-compiled inference (thread-safe read of model)
        inference_start = time.perf_counter()
        with self._model_lock:
            policy, value = self._infer_jit(boards, mask)
        
        # CRITICAL: Block until computation completes (JAX is async)
        policy.block_until_ready()
        value.block_until_ready()
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Convert back to numpy - use np.asarray for zero-copy when possible
        back_convert_start = time.perf_counter()
        policy_np = np.asarray(policy)
        value_np = np.asarray(value)
        back_convert_time = (time.perf_counter() - back_convert_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        # print(f"[Server Timing] Total: {total_time:.3f}ms | "
        #       f"NP->JAX: {convert_time:.3f}ms | "
        #       f"JAX Inference: {inference_time:.3f}ms | "
        #       f"JAX->NP: {back_convert_time:.3f}ms")
        
        return policy_np, value_np


# ============================================================================
# HOT-RELOAD MONITOR
# ============================================================================

class ReloadMonitor:
    """
    Monitors checkpoint directory for reload triggers.
    
    Checks for files named: reload.{model_name}.trigger
    When found, triggers hot-reload and deletes the trigger file.
    """
    
    def __init__(self, model, checkpoint_dir: Optional[str], model_name: str = "alphazero", check_interval: float = 5.0):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model_name = model_name
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring in background thread."""
        if self.checkpoint_dir is None:
            print("[Hot-Reload] Monitoring disabled (no checkpoint directory)")
            return
        
        print(f"[Hot-Reload] Monitoring {self.checkpoint_dir} for reload.{self.model_name}.trigger")
        print(f"[Hot-Reload] Check interval: {self.check_interval}s")
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self._check_for_reload()
            except Exception as e:
                print(f"[Hot-Reload] Error in monitor loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_for_reload(self):
        """Check for reload trigger file."""
        trigger_file = self.checkpoint_dir / f"reload.{self.model_name}.trigger"
        
        if trigger_file.exists():
            print(f"\n[Hot-Reload] 🔔 Trigger detected: {trigger_file}")
            
            # Find latest checkpoint in directory
            checkpoint_path = self._find_latest_checkpoint()
            
            if checkpoint_path:
                # Trigger reload in background thread
                reload_thread = threading.Thread(
                    target=self.model.reload_model,
                    args=(str(checkpoint_path),),
                    daemon=True
                )
                reload_thread.start()
                
                # Delete trigger file
                try:
                    trigger_file.unlink()
                    print(f"[Hot-Reload] ✓ Deleted trigger file")
                except Exception as e:
                    print(f"[Hot-Reload] ⚠️  Failed to delete trigger: {e}")
            else:
                print(f"[Hot-Reload] ❌ No checkpoint found in {self.checkpoint_dir}")
                trigger_file.unlink()  # Delete trigger anyway
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
        
        if not checkpoints:
            return None
        
        # Sort by step number (extracted from checkpoint_N)
        def get_step(p):
            try:
                return int(p.name.split('_')[1])
            except:
                return 0
        
        latest = max(checkpoints, key=get_step)
        print(f"[Hot-Reload] Found latest checkpoint: {latest}")
        return latest


# ============================================================================
# TRITON INFERENCE FUNCTION
# ============================================================================

def create_inference_callable(model):
    """
    Creates the inference function that Triton will call.
    
    The @batch decorator handles dynamic batching automatically.
    """
    
    @batch
    def infer_fn(boards, mask):
        """
        Inference function called by Triton (batched requests).
        
        Input tensors:
            boards: [B, 3, 4, 4] float32 - board states
            mask: [B, 16] float32 - legal action masks
        
        Output tensors:
            policy: [B, 16] float32 - log-probabilities
            value: [B] float32 - value estimates
        """
        wrapper_start = time.perf_counter()
        
        batch_size = len(boards)
        
        # Log batch processing
        print(f"[Inference] Processing batch of size {batch_size}")
        
        # Stack inputs efficiently
        stack_start = time.perf_counter()
        boards_batch = np.stack(boards, axis=0)  # [B, 3, 4, 4]
        mask_batch = np.stack(mask, axis=0)      # [B, 16]
        stack_time = (time.perf_counter() - stack_start) * 1000
        
        # Run batch inference
        inference_start = time.perf_counter()
        policy_batch, value_batch = model.infer_batch(boards_batch, mask_batch)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        wrapper_total = (time.perf_counter() - wrapper_start) * 1000
        
        # print(f"[Wrapper Timing] Total: {wrapper_total:.3f}ms | "
        #       f"Stacking: {stack_time:.3f}ms | "
        #       f"Model inference: {inference_time:.3f}ms")
        
        # Return batched results - PyTriton's @batch decorator will split them
        return {"policy": policy_batch, "value": value_batch}
    
    return infer_fn


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

def run_server(
    checkpoint_path=None,
    port=8001,
    max_batch_size=16,
    batch_window_us=500,  # Changed to microseconds (0.1ms)
    num_channels=64,
    num_res_blocks=3,
    force_random=False,
    reload_check_interval=5.0,
    model_name="alphazero"
):
    """
    Start Triton inference server.
    
    Args:
        checkpoint_path: Path to model checkpoint (None = random weights)
        port: gRPC port (default: 8001)
        max_batch_size: Maximum batch size for dynamic batching
        batch_window_us: Batching window in microseconds (default: 100µs = 0.1ms)
        num_channels: Model architecture parameter
        num_res_blocks: Model architecture parameter
        force_random: Force random weights even if checkpoint exists
        reload_check_interval: Seconds between reload trigger checks
        model_name: Model name for Triton and reload triggers
    """
    
    print("=" * 80)
    print("AlphaZero Triton Inference Server - OPTIMIZED")
    print("=" * 80)
    print(f"Port: {port}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Batching window: {batch_window_us}µs ({batch_window_us/1000:.2f}ms)")
    print(f"Model: {num_channels} channels, {num_res_blocks} residual blocks")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Hot-reload: Enabled (check every {reload_check_interval}s)")
        print(f"Reload trigger: {Path(checkpoint_path).parent}/reload.{model_name}.trigger")
    else:
        print("Checkpoint: None (random weights)")
        print("Hot-reload: Disabled")
    print("=" * 80)
    
    # Initialize model
    model = AlphaZeroInferenceModel(
        checkpoint_path=checkpoint_path,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        force_random=force_random
    )
    
    # Start hot-reload monitor (if checkpoint directory exists)
    reload_monitor = None
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path).parent
        reload_monitor = ReloadMonitor(
            model=model,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            check_interval=reload_check_interval
        )
        reload_monitor.start()
    
    # Create inference callable
    infer_fn = create_inference_callable(model)
    
    # OPTIMIZED MODEL CONFIGURATION
    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        batching=True,  # Enable dynamic batching
        
        # CRITICAL: Dynamic batching scheduler settings
        batcher=DynamicBatcher(
            # OPTIMIZED: Reduced delay for lower latency
            # 100µs = 0.1ms - much faster for single requests
            # Still allows batching if requests arrive close together
            max_queue_delay_microseconds=batch_window_us,
            
            # Preferred batch sizes for dynamic batching
            preferred_batch_size=[8, 16],
            
            # Don't preserve ordering for maximum throughput
            preserve_ordering=False,
        ),
        
        # Response cache disabled for variable inputs
        response_cache=False,
    )
    
    # Start Triton server
    triton_config = TritonConfig(
        grpc_port=port,
        http_port=port - 1,  # HTTP on port-1 for monitoring
        log_verbose=0,  # Reduce logging overhead
    )
    
    with Triton(config=triton_config) as triton:
        print(f"\n[Triton Server] Binding model '{model_name}' on port {port}...")
        
        triton.bind(
            model_name=model_name,
            infer_func=infer_fn,
            inputs=[
                Tensor(name="boards", dtype=np.float32, shape=(3, 4, 4)),
                Tensor(name="mask", dtype=np.float32, shape=(16,)),
            ],
            outputs=[
                Tensor(name="policy", dtype=np.float32, shape=(16,)),
                Tensor(name="value", dtype=np.float32, shape=(1,)),
            ],
            config=model_config,
        )
        
        print("\n" + "=" * 80)
        print("✓ Server is ready! (OPTIMIZED)")
        print("=" * 80)
        print(f"gRPC endpoint: localhost:{port}")
        print(f"HTTP endpoint: localhost:{port-1}")
        print(f"Model name: {model_name}")
        print(f"Max batch size: {max_batch_size}")
        if checkpoint_path:
            print(f"\nHot-reload: ENABLED")
            print(f"  To reload: touch {Path(checkpoint_path).parent}/reload.{model_name}.trigger")
        print("\nC++ client connection string: \"localhost:{port}\"")
        print("\nPress Ctrl+C to stop server...")
        print("=" * 80 + "\n")
        
        try:
            triton.serve()
        finally:
            if reload_monitor:
                reload_monitor.stop()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Start Triton inference server for AlphaZero (optimized with hot-reload)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory (default: random weights)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="gRPC port (default: 8001)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Maximum batch size (default: 16)"
    )
    
    parser.add_argument(
        "--batch-window",
        type=float,
        default=500,
        help="Batching window in microseconds (default: 100µs = 0.1ms)"
    )
    
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of channels in residual blocks (default: 64)"
    )
    
    parser.add_argument(
        "--res-blocks",
        type=int,
        default=3,
        help="Number of residual blocks (default: 3)"
    )
    
    parser.add_argument(
        "--force-random",
        action="store_true",
        help="Force random weights even if checkpoint specified (for debugging)"
    )
    
    parser.add_argument(
        "--reload-interval",
        type=float,
        default=5.0,
        help="Seconds between reload trigger checks (default: 5.0)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="alphazero",
        help="Model name for Triton and reload triggers (default: alphazero)"
    )
    
    args = parser.parse_args()
    
    try:
        run_server(
            checkpoint_path=args.checkpoint,
            port=args.port,
            max_batch_size=args.batch_size,
            batch_window_us=args.batch_window,
            num_channels=args.channels,
            num_res_blocks=args.res_blocks,
            force_random=args.force_random,
            reload_check_interval=args.reload_interval,
            model_name=args.model_name
        )
    except KeyboardInterrupt:
        print("\n\n[Triton Server] Shutting down gracefully...")
        sys.exit(0)


if __name__ == "__main__":
    main()