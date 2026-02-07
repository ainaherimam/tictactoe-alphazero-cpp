#!/usr/bin/env python3
"""
OPTIMIZED TRITON SERVER - MAXIMUM SPEED & ROBUSTNESS
- Dynamic batching with minimal delay
- Optimized batch sizes
- Fast response times
"""

import numpy as np
import jax
import jax.numpy as jnp

from alphazero_model import create_inference_state, AlphaZeroNet
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

class AlphaZeroInferenceModelRANDOM:
    """Model that ONLY uses random weights - optimized for speed"""
    
    def __init__(self, num_channels=64, num_res_blocks=3):
        print("[Server] Initializing with RANDOM WEIGHTS ONLY")
        print("[Server] num_channels:", num_channels)
        print("[Server] num_res_blocks:", num_res_blocks)
        print("[Server] Board: 4x4 (16 actions)")
        
        # FORCE random initialization
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
        
        print("[Server] JIT-compiling...")
        self._infer_jit = jax.jit(self._infer_fn)
        
        # CRITICAL: Comprehensive warmup for all batch sizes
        # This eliminates JIT compilation delays during inference
        print("[Server] Running warmup inferences to trigger JIT compilation...")
        warmup_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
        for batch_size in warmup_sizes:
            dummy_boards = jnp.zeros((batch_size, 3, 4, 4), dtype=jnp.float32)
            dummy_mask = jnp.ones((batch_size, 16), dtype=jnp.float32)
            # Run twice to ensure compilation is complete
            _ = self._infer_jit(dummy_boards, dummy_mask)
            _ = self._infer_jit(dummy_boards, dummy_mask)
            print(f"  ✓ Warmup complete for batch_size={batch_size}")
        
        print("[Server] ✓ Ready with random weights!")
    
    def _infer_fn(self, boards, mask):
        variables = {'params': self.params, 'batch_stats': self.batch_stats}
        return self.apply_fn(variables, boards, mask, training=False)
    
    def infer_batch(self, boards_np, mask_np):
        import time
        total_start = time.perf_counter()
        
        # OPTIMIZED: Use jnp.asarray instead of jnp.array to avoid unnecessary copies
        # asarray will create a view if possible, array always copies
        convert_start = time.perf_counter()
        boards = jnp.asarray(boards_np, dtype=jnp.float32)
        mask = jnp.asarray(mask_np, dtype=jnp.float32)
        convert_time = (time.perf_counter() - convert_start) * 1000
        
        # Run inference
        inference_start = time.perf_counter()
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
        
        print(f"[Server Timing] Total: {total_time:.3f}ms | "
              f"NP->JAX: {convert_time:.3f}ms | "
              f"JAX Inference: {inference_time:.3f}ms | "
              f"JAX->NP: {back_convert_time:.3f}ms")
        
        return policy_np, value_np

def create_inference_callable(model):
    @batch
    def infer_fn(boards, mask):
        import time
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
        
        print(f"[Wrapper Timing] Total: {wrapper_total:.3f}ms | "
              f"Stacking: {stack_time:.3f}ms | "
              f"Model inference: {inference_time:.3f}ms")
        
        # Return batched results - PyTriton handles splitting
        return {"policy": policy_batch, "value": value_batch}
    
    return infer_fn

if __name__ == "__main__":
    print("=" * 80)
    print("AlphaZero Triton Server - OPTIMIZED FOR MAXIMUM SPEED")
    print("=" * 80)
    print("Port: 8001")
    print("Configuration: 64 channels, 3 res blocks, 4x4 board")
    print("Dynamic Batching: ENABLED")
    print("=" * 80)
    
    # Create model with random weights
    model = AlphaZeroInferenceModelRANDOM(num_channels=64, num_res_blocks=3)
    infer_fn = create_inference_callable(model)
    
    # CRITICAL: Configure Triton for optimal performance
    triton_config = TritonConfig(
        grpc_port=8001,
        http_port=8000,  # Also enable HTTP for monitoring
        log_verbose=0,   # Reduce logging overhead
    )
    
    with Triton(config=triton_config) as triton:
        print("\n[Server] Binding model 'alphazero'...")
        
        # OPTIMIZED MODEL CONFIGURATION
        model_config = ModelConfig(
            # Batching configuration - CRITICAL FOR PERFORMANCE
            max_batch_size=16,        # Maximum batch size
            batching=True,            # Enable dynamic batching
            
            # CRITICAL: Dynamic batching scheduler settings
            # Configure the DynamicBatcher object with proper parameters
            batcher=DynamicBatcher(
                # OPTIMIZED: Reduced delay for lower latency
                # 100µs = 0.1ms - much faster for single requests
                # Still allows batching if requests arrive close together
                max_queue_delay_microseconds=0.1,  # 0.1ms max wait
                
                # Preferred batch sizes for dynamic batching
                # Triton will try to create batches of these sizes
                preferred_batch_size=[4, 8, 16],
                
                # Preserve request ordering (optional, adds slight overhead)
                preserve_ordering=False,
            ),
            
            # Response cache disabled for variable inputs
            response_cache=False,
        )
        
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
        print("✓ Server ready! (OPTIMIZED)")
        print("=" * 80)
        print("gRPC: localhost:8001")
        print("HTTP: localhost:8000")
        print("Max batch size: 16")
        print("\nPress Ctrl+C to stop")
        print("=" * 80 + "\n")
        
        triton.serve()