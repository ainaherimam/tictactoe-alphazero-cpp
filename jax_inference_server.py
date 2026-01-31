#!/usr/bin/env python3
"""
JAX Inference Server for MCTS with Shared Memory

This server handles neural network inference for MCTS using JAX.
It communicates with C++ via shared memory for maximum performance.

Usage:
    python3 jax_inference_server.py --model model1.pkl --model2 model2.pkl
"""

import jax
import jax.numpy as jnp
import numpy as np
from multiprocessing import shared_memory
import struct
import time
import signal
import sys
import argparse
from typing import Tuple, Optional
import pickle

# ============================================================================
# CONFIGURATION - Must match C++ side!
# ============================================================================
MAX_BATCH_SIZE = 256
INPUT_CHANNELS = 3
BOARD_HEIGHT = 4
BOARD_WIDTH = 4
INPUT_SIZE = INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH
POLICY_SIZE = 16

# ============================================================================
# Memory Layout Offsets (must match C++ SharedMemoryBuffer)
# ============================================================================
WRITE_INDEX_OFFSET = 0
READ_INDEX_OFFSET = 4
PROCESSED_INDEX_OFFSET = 8
SHUTDOWN_OFFSET = 12
SERVER_READY_OFFSET = 13

# Requests start after 4KB control block
REQUESTS_OFFSET = 4096

# Calculate request stride (64-byte aligned)
REQUEST_SIZE = INPUT_SIZE * 4 + POLICY_SIZE * 4 + 4 + 4 + 1  # input + mask + req_id + model_id + ready
REQUEST_STRIDE = ((REQUEST_SIZE + 63) // 64) * 64  # Round up to 64-byte alignment

# Responses come after all requests
RESPONSES_OFFSET = REQUESTS_OFFSET + MAX_BATCH_SIZE * REQUEST_STRIDE
RESPONSE_SIZE = POLICY_SIZE * 4 + 4 + 4  # policy + value + request_id
RESPONSE_STRIDE = ((RESPONSE_SIZE + 63) // 64) * 64

# Statistics at the end
STATS_OFFSET = RESPONSES_OFFSET + MAX_BATCH_SIZE * RESPONSE_STRIDE


class JAXInferenceServer:
    """
    Shared memory-based inference server for JAX models
    """
    
    def __init__(self, 
                 model1_apply_fn,
                 model1_params,
                 model2_apply_fn=None,
                 model2_params=None,
                 shm_name: str = "mcts_jax_inference",
                 batch_timeout_ms: float = 5.0,
                 device: str = 'cpu'):
        """
        Initialize JAX inference server
        
        Args:
            model1_apply_fn: JAX function for model 1 inference
            model1_params: Parameters for model 1
            model2_apply_fn: JAX function for model 2 inference (optional)
            model2_params: Parameters for model 2 (optional)
            shm_name: Shared memory name
            batch_timeout_ms: Max time to wait for batch to fill (ms)
            device: 'gpu' or 'cpu'
        """
        self.model1_apply = model1_apply_fn
        self.model1_params = model1_params
        self.model2_apply = model2_apply_fn
        self.model2_params = model2_params
        
        self.shm_name = shm_name
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.device = device
        
        # Calculate total buffer size
        self.buffer_size = self._calculate_buffer_size()
        
        print(f"[JAX Server] Initializing...")
        print(f"[JAX Server] Buffer size: {self.buffer_size / 1024:.1f} KB")
        print(f"[JAX Server] Max batch size: {MAX_BATCH_SIZE}")
        print(f"[JAX Server] Input size: {INPUT_SIZE}")
        print(f"[JAX Server] Policy size: {POLICY_SIZE}")
        
        # Connect to shared memory
        self._connect_shared_memory()
        
        # JIT compile inference functions
        print("[JAX Server] JIT compiling inference functions...")
        self._compile_inference()
        print("[JAX Server] ✓ JIT compilation complete")
        
        # Statistics
        self.total_batches = 0
        self.total_requests = 0
        self.last_read_index = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        # Mark server as ready
        self.shm.buf[SERVER_READY_OFFSET] = 1
        print("[JAX Server] ✓ Server ready!\n")
    
    def _calculate_buffer_size(self) -> int:
        """Calculate total shared memory size needed"""
        control_block = 4096
        requests_size = MAX_BATCH_SIZE * REQUEST_STRIDE
        responses_size = MAX_BATCH_SIZE * RESPONSE_STRIDE
        stats_size = 64
        return control_block + requests_size + responses_size + stats_size
    
    def _connect_shared_memory(self):
        """Connect to or create shared memory"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            print(f"[JAX Server] ✓ Connected to existing shared memory: {self.shm_name}")
        except FileNotFoundError:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.buffer_size
            )
            print(f"[JAX Server] ✓ Created new shared memory: {self.shm_name}")
            # Zero out the buffer
            self.shm.buf[:self.buffer_size] = b'\x00' * self.buffer_size
    
    def _compile_inference(self):
        """JIT compile inference functions with warmup"""
        # Compile model 1
        dummy_input = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        dummy_mask = jnp.ones((1, POLICY_SIZE))
        
        self.batched_inference_model1 = jax.jit(
            lambda params, x, m: self._inference_with_mask(
                self.model1_apply, params, x, m
            )
        )
        
        # Warmup
        _ = self.batched_inference_model1(self.model1_params, dummy_input, dummy_mask)
        print("[JAX Server] ✓ Model 1 compiled")
        
        # Compile model 2 if provided
        if self.model2_apply is not None and self.model2_params is not None:
            self.batched_inference_model2 = jax.jit(
                lambda params, x, m: self._inference_with_mask(
                    self.model2_apply, params, x, m
                )
            )
            _ = self.batched_inference_model2(self.model2_params, dummy_input, dummy_mask)
            print("[JAX Server] ✓ Model 2 compiled")
        else:
            self.batched_inference_model2 = None
    
    def _inference_with_mask(self, apply_fn, params, inputs, masks):
        """
        Run inference and apply legal move masking
        
        Args:
            apply_fn: Model apply function
            params: Model parameters
            inputs: Input tensor [batch, C, H, W]
            masks: Legal move masks [batch, A]
        
        Returns:
            (policy, value) where policy is masked and normalized
        """
        policy_logits, value = apply_fn(params, inputs)
        
        # Apply legal move mask (set illegal moves to -inf)
        policy_logits = jnp.where(masks > 0.5, policy_logits, -1e9)
        
        # Softmax to get probabilities
        policy = jax.nn.softmax(policy_logits, axis=-1)
        
        return policy, value
    
    def run(self):
        """Main server loop - processes inference requests"""
        print("[JAX Server] Starting main loop...")
        print(f"[JAX Server] Batch timeout: {self.batch_timeout * 1000:.1f}ms")
        print(f"[JAX Server] Device: {self.device}\n")
        
        while not self._should_shutdown():
            # Read current write index
            write_idx = struct.unpack_from('I', self.shm.buf, WRITE_INDEX_OFFSET)[0]
            
            if write_idx == self.last_read_index:
                # No new requests, sleep briefly
                time.sleep(0.0001)  # 100 microseconds
                continue
            
            # Collect batch
            batch_start_time = time.time()
            batch_inputs_m1 = []
            batch_masks_m1 = []
            batch_indices_m1 = []
            
            batch_inputs_m2 = []
            batch_masks_m2 = []
            batch_indices_m2 = []
            
            requests_read = 0
            
            # Read up to MAX_BATCH_SIZE requests or until timeout
            while requests_read < MAX_BATCH_SIZE:
                current_write = struct.unpack_from('I', self.shm.buf, WRITE_INDEX_OFFSET)[0]
                
                if self.last_read_index >= current_write:
                    # No more requests available
                    break
                
                # Check timeout
                if time.time() - batch_start_time > self.batch_timeout:
                    break
                
                # Read request at current position
                req_idx = self.last_read_index % MAX_BATCH_SIZE
                req_offset = REQUESTS_OFFSET + req_idx * REQUEST_STRIDE
                
                # Extract input (flattened [C*H*W])
                input_data = np.frombuffer(
                    self.shm.buf,
                    dtype=np.float32,
                    count=INPUT_SIZE,
                    offset=req_offset
                ).reshape(INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).copy()
                
                # Extract mask
                mask_data = np.frombuffer(
                    self.shm.buf,
                    dtype=np.float32,
                    count=POLICY_SIZE,
                    offset=req_offset + INPUT_SIZE * 4
                ).copy()
                
                # Extract metadata
                request_id = struct.unpack_from('I', self.shm.buf, 
                                               req_offset + (INPUT_SIZE + POLICY_SIZE) * 4)[0]
                model_id = struct.unpack_from('I', self.shm.buf,
                                              req_offset + (INPUT_SIZE + POLICY_SIZE) * 4 + 4)[0]
                
                # Route to appropriate model
                if model_id == 0:
                    batch_inputs_m1.append(input_data)
                    batch_masks_m1.append(mask_data)
                    batch_indices_m1.append((req_idx, request_id))
                else:
                    batch_inputs_m2.append(input_data)
                    batch_masks_m2.append(mask_data)
                    batch_indices_m2.append((req_idx, request_id))
                
                self.last_read_index += 1
                requests_read += 1
            
            # Update read index
            struct.pack_into('I', self.shm.buf, READ_INDEX_OFFSET, self.last_read_index)
            
            # Process model 1 batch
            if batch_inputs_m1:
                self._process_batch(
                    batch_inputs_m1, 
                    batch_masks_m1, 
                    batch_indices_m1,
                    self.batched_inference_model1,
                    self.model1_params,
                    model_id=0
                )
            
            # Process model 2 batch
            if batch_inputs_m2 and self.batched_inference_model2 is not None:
                self._process_batch(
                    batch_inputs_m2,
                    batch_masks_m2,
                    batch_indices_m2,
                    self.batched_inference_model2,
                    self.model2_params,
                    model_id=1
                )
        
        print("\n[JAX Server] Shutting down gracefully...")
        self._cleanup()
    
    def _process_batch(self, inputs, masks, indices, inference_fn, params, model_id):
        """Process a batch of requests"""
        batch_size = len(inputs)
        
        # Convert to JAX arrays
        inputs_array = jnp.array(inputs)
        masks_array = jnp.array(masks)
        
        # Run inference
        policies, values = inference_fn(params, inputs_array, masks_array)
        
        # Write results back to shared memory
        for i, (req_idx, request_id) in enumerate(indices):
            resp_offset = RESPONSES_OFFSET + req_idx * RESPONSE_STRIDE
            
            # Write policy
            policy_bytes = np.array(policies[i], dtype=np.float32).tobytes()
            self.shm.buf[resp_offset:resp_offset + POLICY_SIZE * 4] = policy_bytes
            
            # Write value
            value_bytes = struct.pack('f', float(values[i]))
            self.shm.buf[resp_offset + POLICY_SIZE * 4:
                        resp_offset + POLICY_SIZE * 4 + 4] = value_bytes
            
            # Write request_id
            struct.pack_into('I', self.shm.buf,
                           resp_offset + (POLICY_SIZE + 1) * 4, request_id)
            
            # Mark as ready (atomic flag in request)
            ready_offset = REQUESTS_OFFSET + req_idx * REQUEST_STRIDE + (INPUT_SIZE + POLICY_SIZE + 2) * 4
            self.shm.buf[ready_offset] = 1
        
        # Update statistics
        self.total_batches += 1
        self.total_requests += batch_size
        struct.pack_into('Q', self.shm.buf, STATS_OFFSET + 8, self.total_batches)
        
        # Periodic logging
        if self.total_batches % 100 == 0:
            avg_batch = self.total_requests / self.total_batches
            print(f"[JAX Server] Processed {self.total_batches} batches "
                  f"({self.total_requests} requests, avg={avg_batch:.1f}, "
                  f"model={model_id})")
    
    def _should_shutdown(self) -> bool:
        """Check if shutdown flag is set"""
        return struct.unpack_from('?', self.shm.buf, SHUTDOWN_OFFSET)[0]
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n[JAX Server] Received shutdown signal")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Clean up shared memory"""
        print(f"[JAX Server] Final stats: {self.total_batches} batches, "
              f"{self.total_requests} requests")
        self.shm.close()
        try:
            self.shm.unlink()
            print("[JAX Server] ✓ Shared memory cleaned up")
        except:
            pass


# ============================================================================
# Example JAX Model (replace with your actual model)
# ============================================================================

def create_example_model():
    """
    Example model structure - REPLACE THIS with your actual JAX model
    
    Your model should have a function signature like:
        def apply(params, x) -> (policy_logits, value)
    
    Where:
        x: [batch, channels, height, width]
        policy_logits: [batch, action_space]
        value: [batch]
    """
    import flax.linen as nn
    
    class AlphaZeroNet(nn.Module):
        num_actions: int
        
        @nn.compact
        def __call__(self, x):
            # Shared representation
            x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x)
            
            # Residual blocks
            for _ in range(3):
                residual = x
                x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
                x = nn.BatchNorm(use_running_average=True)(x)
                x = nn.relu(x)
                x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
                x = nn.BatchNorm(use_running_average=True)(x)
                x = x + residual
                x = nn.relu(x)
            
            # Policy head
            p = nn.Conv(features=2, kernel_size=(1, 1))(x)
            p = nn.BatchNorm(use_running_average=True)(p)
            p = nn.relu(p)
            p = p.reshape((p.shape[0], -1))
            policy_logits = nn.Dense(features=self.num_actions)(p)
            
            # Value head
            v = nn.Conv(features=1, kernel_size=(1, 1))(x)
            v = nn.BatchNorm(use_running_average=True)(v)
            v = nn.relu(v)
            v = v.reshape((v.shape[0], -1))
            v = nn.Dense(features=64)(v)
            v = nn.relu(v)
            value = nn.Dense(features=1)(v)
            value = jnp.tanh(value).squeeze(-1)
            
            return policy_logits, value
    
    # Initialize model
    model = AlphaZeroNet(num_actions=POLICY_SIZE)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    params = model.init(rng, dummy_input)
    
    return model.apply, params


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='JAX Inference Server for MCTS')
    parser.add_argument('--model1', type=str, help='Path to model 1 checkpoint (.pkl)')
    parser.add_argument('--model2', type=str, help='Path to model 2 checkpoint (.pkl)')
    parser.add_argument('--shm-name', type=str, default='mcts_jax_inference',
                       help='Shared memory name')
    parser.add_argument('--batch-timeout', type=float, default=5.0,
                       help='Batch collection timeout (ms)')
    parser.add_argument('--device', type=str, default='cpu', choices=['gpu', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load or create models
    if args.model1:
        print(f"[JAX Server] Loading model 1 from {args.model1}")
        with open(args.model1, 'rb') as f:
            model1_data = pickle.load(f)
        model1_apply = model1_data['apply_fn']
        model1_params = model1_data['params']
    else:
        print("[JAX Server] Using example model 1")
        model1_apply, model1_params = create_example_model()
    
    if args.model2:
        print(f"[JAX Server] Loading model 2 from {args.model2}")
        with open(args.model2, 'rb') as f:
            model2_data = pickle.load(f)
        model2_apply = model2_data['apply_fn']
        model2_params = model2_data['params']
    else:
        print("[JAX Server] No model 2 specified")
        model2_apply = None
        model2_params = None
    
    # Create and run server
    server = JAXInferenceServer(
        model1_apply_fn=model1_apply,
        model1_params=model1_params,
        model2_apply_fn=model2_apply,
        model2_params=model2_params,
        shm_name=args.shm_name,
        batch_timeout_ms=args.batch_timeout,
        device=args.device
    )
    
    server.run()


if __name__ == "__main__":
    main()