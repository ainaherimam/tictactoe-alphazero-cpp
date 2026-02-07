#!/usr/bin/env python3
"""
Triton Inference Server for AlphaZero
======================================

Serves JAX model via gRPC with dynamic batching for low-latency inference.

Requirements:
    pip install nvidia-pytriton jax[cpu] flax optax orbax-checkpoint --break-system-packages

Usage:
    python triton_server.py [--checkpoint CKPT_PATH] [--port 8001] [--batch-size 16]

Features:
    - JIT-compiled JAX model (loaded once, stays in memory)
    - Dynamic batching (accumulates requests, batches inference)
    - Async gRPC interface for C++ clients
    - <2ms inference latency with batching
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

# Import your AlphaZero model
from alphazero_model import (
    load_checkpoint_for_inference,
    create_inference_state,
    AlphaZeroNet
)

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


# ============================================================================
# MODEL WRAPPER (JIT-COMPILED FOR PERFORMANCE)
# ============================================================================

class AlphaZeroInferenceModel:
    """
    Wraps JAX model for Triton inference.
    
    - Loads checkpoint or initializes random weights
    - JIT-compiles inference function on first call
    - Handles batched inputs efficiently
    """
    
    def __init__(self, checkpoint_path=None, num_channels=64, num_res_blocks=3, force_random=False):
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        
        print("[Triton Server] Initializing AlphaZero model...")
        
        # FORCE random weights for debugging
        if force_random:
            print("[Triton Server] ⚠️  FORCE_RANDOM_WEIGHTS enabled - ignoring any checkpoint!")
            checkpoint_path = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Triton Server] Loading checkpoint from {checkpoint_path}")
            state = load_checkpoint_for_inference(
                checkpoint_path=checkpoint_path,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                num_actions=16
            )
        else:
            if checkpoint_path:
                print(f"[Triton Server] ⚠️  Checkpoint not found: {checkpoint_path}")
            print("[Triton Server] Initializing with random weights")
            rng = jax.random.PRNGKey(42)
            state = create_inference_state(
                rng=rng,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
                num_actions=16
            )
        
        self.params = state['params']
        self.batch_stats = state['batch_stats']
        self.apply_fn = state['apply_fn']
        
        # JIT-compile the inference function
        print("[Triton Server] JIT-compiling inference function...")
        self._infer_jit = jax.jit(self._infer_fn)
        
        # Comprehensive warmup - compile for all common batch sizes
        print("[Triton Server] Running warmup inferences...")
        for batch_size in [1, 2, 4, 8, 16]:
            dummy_boards = jnp.zeros((batch_size, 3, 4, 4), dtype=jnp.float32)
            dummy_mask = jnp.ones((batch_size, 16), dtype=jnp.float32)
            _ = self._infer_jit(dummy_boards, dummy_mask)
            print(f"[Triton Server]   ✓ Batch size {batch_size} ready")
        
        print("[Triton Server] ✓ Model ready (JIT-compiled and cached for all batch sizes)")
    
    def _infer_fn(self, boards, mask):
        """
        Core inference function (will be JIT-compiled).
        
        Args:
            boards: [B, 3, 4, 4] board states
            mask: [B, 16] legal action masks
        
        Returns:
            policy: [B, 16] log-probabilities
            value: [B] value estimates
        """
        variables = {
            'params': self.params,
            'batch_stats': self.batch_stats
        }
        
        policy, value = self.apply_fn(
            variables,
            boards,
            mask,
            training=False
        )
        
        return policy, value
    
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
        # Convert to JAX arrays
        boards = jnp.array(boards_np, dtype=jnp.float32)
        mask = jnp.array(mask_np, dtype=jnp.float32)
        
        # Run JIT-compiled inference
        policy, value = self._infer_jit(boards, mask)
        
        # Convert back to numpy
        return np.array(policy), np.array(value)


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
        # boards and mask are lists of numpy arrays (one per request in batch)
        # Each array already has the correct shape: (3, 4, 4) and (16,)
        # Stack them into a single batch
        boards_batch = np.stack(boards, axis=0)  # [B, 3, 4, 4]
        mask_batch = np.stack(mask, axis=0)      # [B, 16]
        
        # Run batched inference
        policy_batch, value_batch = model.infer_batch(boards_batch, mask_batch)
        
        # Return batched numpy arrays - PyTriton's @batch decorator will split them
        return {"policy": policy_batch, "value": value_batch}
    
    return infer_fn


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

def run_server(
    checkpoint_path=None,
    port=8001,
    max_batch_size=16,
    batch_window_ms=5,
    num_channels=64,
    num_res_blocks=3,
    force_random=False
):
    """
    Start Triton inference server.
    
    Args:
        checkpoint_path: Path to model checkpoint (None = random weights)
        port: gRPC port (default: 8001)
        max_batch_size: Maximum batch size for dynamic batching
        batch_window_ms: Time window to accumulate requests (ms)
        num_channels: Model architecture parameter
        num_res_blocks: Model architecture parameter
    """
    
    print("=" * 80)
    print("AlphaZero Triton Inference Server")
    print("=" * 80)
    print(f"Port: {port}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Batching window: {batch_window_ms}ms")
    print(f"Model: {num_channels} channels, {num_res_blocks} residual blocks")
    print("=" * 80)
    
    # Initialize model
    model = AlphaZeroInferenceModel(
        checkpoint_path=checkpoint_path,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        force_random=force_random
    )
    
    # Create inference callable
    infer_fn = create_inference_callable(model)
    
    # Configure model for Triton
    # Note: Using minimal config for compatibility across PyTriton versions
    # Dynamic batching is enabled automatically with max_batch_size > 1
    # PyTriton will accumulate requests for a short time window before inference
    # The batch_window_ms parameter is informational for logging only
    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        batching=True,  # Enable dynamic batching
    )
    
    # Start Triton server
    triton_config = TritonConfig(grpc_port=port)
    with Triton(config=triton_config) as triton:
        print(f"\n[Triton Server] Binding model 'alphazero' on port {port}...")
        
        triton.bind(
            model_name="alphazero",
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
        print("✓ Server is ready!")
        print("=" * 80)
        print(f"gRPC endpoint: localhost:{port}")
        print(f"Model name: alphazero")
        print("\nC++ client connection string: \"localhost:{port}\"")
        print("\nPress Ctrl+C to stop server...")
        print("=" * 80 + "\n")
        
        triton.serve()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Start Triton inference server for AlphaZero"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: random weights for 4x4 board)"
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
        type=int,
        default=5,
        help="Batching window in milliseconds (default: 5)"
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
    
    args = parser.parse_args()
    
    try:
        run_server(
            checkpoint_path=args.checkpoint,
            port=args.port,
            max_batch_size=args.batch_size,
            batch_window_ms=args.batch_window,
            num_channels=args.channels,
            num_res_blocks=args.res_blocks,
            force_random=args.force_random
        )
    except KeyboardInterrupt:
        print("\n\n[Triton Server] Shutting down gracefully...")
        sys.exit(0)


if __name__ == "__main__":
    main()