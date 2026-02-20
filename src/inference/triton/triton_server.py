#!/usr/bin/env python3
"""
Triton Inference Server for AlphaZero

Serves JAX model via HTTP/gRPC with dynamic batching for low-latency inference.

Behaviour is identical to random_o.py:
  - Flat @batch infer_fn (no stacking wrapper, no class dispatch)
  - jit_predict receives params/batch_stats as explicit arguments (hot-reload safe)
  - Policy output is log-probs -> converted to probs via jnp.exp() before returning
  - TimingStats with per-batch and rolling-average logging
  - Hot-reload via glob pattern: reload.triton.<checkpoint_number>.trigger
  - All ports / batch config read from env vars

Usage:
    python triton_server.py
    CHECKPOINT_PATH=checkpoints/checkpoint_5100 python triton_server.py

Hot-Reload:
    touch <watch_dir>/reload.triton.<checkpoint_number>.trigger
    e.g.  touch checkpoints/reload.triton.5200.trigger
"""

import glob
import logging
import os
import sys
import time
from threading import Thread

import numpy as np
import jax
import jax.numpy as jnp

from pytriton.decorators import batch
from pytriton.model_config import Tensor, ModelConfig, DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from src.models.alphazero_model import create_inference_state, load_checkpoint_for_inference

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARNING').upper()
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.propagate = False

# ============================================================================
# TIMING STATS
# ============================================================================

class TimingStats:
    def __init__(self):
        self.total_requests = 0
        self.total_prep_time = 0.0
        self.total_inference_time = 0.0
        self.total_postprocess_time = 0.0
        self.total_overhead_time = 0.0

    def add(self, prep, inference, postprocess, overhead):
        self.total_requests += 1
        self.total_prep_time += prep
        self.total_inference_time += inference
        self.total_postprocess_time += postprocess
        self.total_overhead_time += overhead

    def get_averages(self):
        if self.total_requests == 0:
            return 0, 0, 0, 0
        n = self.total_requests
        return (
            self.total_prep_time / n,
            self.total_inference_time / n,
            self.total_postprocess_time / n,
            self.total_overhead_time / n
        )

timing_stats = TimingStats()

# ============================================================================
# GLOBAL STATE
# ============================================================================

MODEL_STATE = None   # dict with 'params', 'batch_stats', 'apply_fn'
jit_predict = None   # jax.jit(lambda params, batch_stats, boards, mask: ...)
request_counter = 0
batch_counter = 0

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model(checkpoint_path=None, num_channels=64, num_res_blocks=3, num_actions=16):
    """Load (or re-load) model weights and (re-)JIT compile inference function."""
    global MODEL_STATE, jit_predict

    logger.info("Loading model...")

    if checkpoint_path and os.path.exists(checkpoint_path):
        MODEL_STATE = load_checkpoint_for_inference(
            checkpoint_path=checkpoint_path,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_actions=num_actions,
        )
    else:
        rng = jax.random.PRNGKey(0)
        MODEL_STATE = create_inference_state(
            rng=rng,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            num_actions=num_actions,
        )

    # Pass params/batch_stats as explicit args so hot-reload works without
    # retracing — a new call with new params reuses the compiled XLA graph.
    jit_predict = jax.jit(
        lambda params, batch_stats, boards, mask: MODEL_STATE['apply_fn'](
            {'params': params, 'batch_stats': batch_stats},
            boards,
            mask,
            training=False,
        )
    )

    # Warmup
    logger.info("Warming up...")
    dummy_boards = jnp.ones((1, 3, 4, 4), dtype=jnp.float32)
    dummy_mask   = jnp.ones((1, num_actions), dtype=jnp.float32)
    _ = jit_predict(MODEL_STATE['params'], MODEL_STATE['batch_stats'], dummy_boards, dummy_mask)

    logger.info("Model ready")

# ============================================================================
# HOT RELOAD
# ============================================================================

def watch_for_reload(watch_dir, num_channels, num_res_blocks, num_actions):
    """
    Watch for reload trigger files of the form:
        reload.triton.<checkpoint_number>.trigger

    On detection, calls setup_model() with the matching checkpoint path, then
    removes the trigger file.
    """
    logger.info(f"Hot reload enabled, watching: {watch_dir}")

    while True:
        try:
            trigger_files = glob.glob(os.path.join(watch_dir, "reload.triton.*.trigger"))

            for trigger_file in trigger_files:
                try:
                    filename = os.path.basename(trigger_file)
                    parts = filename.split('.')
                    if len(parts) >= 3:
                        checkpoint_number = parts[2]
                        checkpoint_path = os.path.join(watch_dir, f"checkpoint_{checkpoint_number}")

                        if os.path.exists(checkpoint_path):
                            logger.info(f"🔄 Hot reload triggered: {checkpoint_path}")
                            setup_model(checkpoint_path, num_channels, num_res_blocks, num_actions)
                            logger.info(f"✓ Reloaded checkpoint_{checkpoint_number}")
                        else:
                            logger.warning(f"Checkpoint path not found: {checkpoint_path}")

                    os.remove(trigger_file)
                    logger.info(f"Removed trigger: {filename}")

                except Exception as e:
                    logger.error(f"Error processing trigger {trigger_file}: {e}")

            time.sleep(1)

        except Exception as e:
            logger.error(f"Hot reload watch error: {e}")
            time.sleep(5)

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

@batch
def infer_fn(boards, mask):
    """
    Ultra-optimized inference with detailed timing.

    PyTriton passes pre-stacked numpy arrays:
        boards: [batch, 48]   float32 — flat board representation
        mask:   [batch, 16]   float32 — legal action mask

    Returns:
        policy: [batch, 16]   float32 — action probabilities (NOT log-probs)
        value:  [batch, 1]    float32 — value estimate
    """
    global request_counter, batch_counter
    total_start = time.perf_counter()
    batch_counter += 1
    batch_size = boards.shape[0]
    request_counter += batch_size

    # -------------------------------------------------------------------------
    # TIMING 1: Data Preparation
    # -------------------------------------------------------------------------
    prep_start = time.perf_counter()

    # Reshape boards: [batch, 48] -> [batch, 3, 4, 4]
    boards_reshaped = boards.reshape(batch_size, 3, 4, 4)

    # Transfer to JAX device
    boards_jax = jnp.asarray(boards_reshaped, dtype=jnp.float32)
    mask_jax   = jnp.asarray(mask,            dtype=jnp.float32)

    prep_time = (time.perf_counter() - prep_start) * 1000  # ms

    # -------------------------------------------------------------------------
    # TIMING 2: Inference
    # -------------------------------------------------------------------------
    inference_start = time.perf_counter()

    log_policy, value = jit_predict(
        MODEL_STATE['params'],
        MODEL_STATE['batch_stats'],
        boards_jax,
        mask_jax,
    )

    # Force synchronization before stopping the clock
    log_policy.block_until_ready()

    inference_time = (time.perf_counter() - inference_start) * 1000  # ms

    # -------------------------------------------------------------------------
    # TIMING 3: Post-processing — convert log-probs -> probs
    # -------------------------------------------------------------------------
    postprocess_start = time.perf_counter()

    policy_np = np.asarray(jnp.exp(log_policy))
    value_np  = np.asarray(value).reshape(batch_size, 1)

    postprocess_time = (time.perf_counter() - postprocess_start) * 1000  # ms

    # -------------------------------------------------------------------------
    # Overhead & logging
    # -------------------------------------------------------------------------
    total_time    = (time.perf_counter() - total_start) * 1000  # ms
    overhead_time = total_time - (prep_time + inference_time + postprocess_time)

    timing_stats.add(prep_time, inference_time, postprocess_time, overhead_time)

    logger.info(
        f"Batch#{batch_counter} size={batch_size} | "
        f"prep={prep_time:.3f}ms "
        f"inference={inference_time:.3f}ms "
        f"postprocess={postprocess_time:.3f}ms "
        f"overhead={overhead_time:.3f}ms "
        f"total={total_time:.3f}ms"
    )

    if batch_counter % 10 == 0:
        avg_prep, avg_inf, avg_post, avg_overhead = timing_stats.get_averages()
        avg_total = avg_prep + avg_inf + avg_post + avg_overhead
        logger.info(
            f"Averages (n={timing_stats.total_requests}): "
            f"prep={avg_prep:.3f}ms "
            f"inference={avg_inf:.3f}ms "
            f"postprocess={avg_post:.3f}ms "
            f"overhead={avg_overhead:.3f}ms "
            f"total={avg_total:.3f}ms"
        )

    return [policy_np, value_np]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the Triton server — all config via argparse."""
    import argparse
    parser = argparse.ArgumentParser(
        description="PyTriton AlphaZero Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint',      type=str, default='checkpoints/checkpoint_5100', help='Path to model checkpoint')
    parser.add_argument('--watch-dir',       type=str, default=None,  help='Directory to watch for reload triggers (default: checkpoint dir)')
    parser.add_argument('--num-channels',    type=int, default=64,    help='Model channels')
    parser.add_argument('--num-res-blocks',  type=int, default=3,     help='Model residual blocks')
    parser.add_argument('--num-actions',     type=int, default=16,    help='Number of actions')
    parser.add_argument('--max-batch-size',  type=int, default=32,    help='Max dynamic batch size')
    parser.add_argument('--no-hot-reload',   action='store_true',     help='Disable hot reload watcher')
    parser.add_argument('--max-queue-delay', type=int, default=500,   help='Dynamic batcher queue delay (us)')
    parser.add_argument('--http-port',       type=int, default=8000,  help='Triton HTTP port')
    parser.add_argument('--grpc-port',       type=int, default=8001,  help='Triton gRPC port')
    parser.add_argument('--metrics-port',    type=int, default=8002,  help='Triton metrics port')
    parser.add_argument('--log-verbose',     type=int, default=0,     help='Triton log verbosity')
    args = parser.parse_args()

    checkpoint_path   = args.checkpoint
    watch_dir         = args.watch_dir if args.watch_dir else checkpoint_path
    num_channels      = args.num_channels
    num_res_blocks    = args.num_res_blocks
    num_actions       = args.num_actions
    max_batch_size    = args.max_batch_size
    enable_hot_reload = not args.no_hot_reload

    # Batching
    max_queue_delay_us    = args.max_queue_delay
    preferred_batch_sizes = [8, 16]

    # Triton ports
    http_port    = args.http_port
    grpc_port    = args.grpc_port
    metrics_port = args.metrics_port
    log_verbose  = args.log_verbose

    logger.info("=" * 60)
    logger.info("PyTriton Server - ULTRA OPTIMIZED (WITH TIMING)")
    logger.info("=" * 60)
    logger.info(f"Log level:             {LOG_LEVEL}")
    logger.info(f"HTTP port:             {http_port}")
    logger.info(f"gRPC port:             {grpc_port}")
    logger.info(f"Metrics port:          {metrics_port}")
    logger.info(f"Max batch size:        {max_batch_size}")
    logger.info(f"Max queue delay:       {max_queue_delay_us}us ({max_queue_delay_us/1000:.2f}ms)")
    logger.info(f"Preferred batch sizes: {preferred_batch_sizes}")
    logger.info(f"Hot reload:            {enable_hot_reload}")
    if enable_hot_reload and watch_dir:
        logger.info(f"Watch directory:       {watch_dir}")

    setup_model(checkpoint_path, num_channels, num_res_blocks, num_actions)

    # Start hot reload watcher thread
    if enable_hot_reload and watch_dir:
        reload_thread = Thread(
            target=watch_for_reload,
            args=(watch_dir, num_channels, num_res_blocks, num_actions),
            daemon=True,
        )
        reload_thread.start()

    # Triton server config
    triton_config = TritonConfig(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        log_verbose=log_verbose,
    )

    batcher = DynamicBatcher(
        max_queue_delay_microseconds=max_queue_delay_us,
        preferred_batch_size=preferred_batch_sizes,
        preserve_ordering=False,
    )

    model_config = ModelConfig(
        batching=True,
        max_batch_size=max_batch_size,
        batcher=batcher,
        response_cache=False,
    )

    triton = Triton(config=triton_config)

    triton.bind(
        model_name="AlphaZero",
        infer_func=infer_fn,
        inputs=[
            Tensor(name="boards", dtype=np.float32, shape=(48,)),
            Tensor(name="mask",   dtype=np.float32, shape=(16,)),
        ],
        outputs=[
            Tensor(name="policy", dtype=np.float32, shape=(16,)),
            Tensor(name="value",  dtype=np.float32, shape=(1,)),
        ],
        config=model_config,
    )

    logger.info("")
    logger.info(f"Server running on localhost:{http_port} (HTTP) and localhost:{grpc_port} (gRPC)")
    logger.info("Dynamic Batcher Configuration:")
    logger.info(f"  max_queue_delay:       {max_queue_delay_us}us ({max_queue_delay_us/1000:.2f}ms)")
    logger.info(f"  max_batch_size:        {max_batch_size}")
    logger.info(f"  preserve_ordering:     False")
    logger.info(f"  preferred_batch_sizes: {preferred_batch_sizes}")
    logger.info("")
    logger.info("Enable detailed timing logs with: export LOG_LEVEL=INFO")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        triton.serve()
    except KeyboardInterrupt:
        logger.info("")
        logger.info(f"Shutdown - Processed {batch_counter} batches, {request_counter} requests")

        if timing_stats.total_requests > 0:
            avg_prep, avg_inf, avg_post, avg_overhead = timing_stats.get_averages()
            avg_total = avg_prep + avg_inf + avg_post + avg_overhead
            logger.info("=" * 60)
            logger.info("FINAL TIMING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total requests:        {timing_stats.total_requests}")
            logger.info(f"Avg preparation:       {avg_prep:.3f} ms ({100*avg_prep/avg_total:.1f}%)")
            logger.info(f"Avg inference:         {avg_inf:.3f} ms ({100*avg_inf/avg_total:.1f}%)")
            logger.info(f"Avg postprocess:       {avg_post:.3f} ms ({100*avg_post/avg_total:.1f}%)")
            logger.info(f"Avg PyTriton overhead: {avg_overhead:.3f} ms ({100*avg_overhead/avg_total:.1f}%)")
            logger.info(f"Avg total:             {avg_total:.3f} ms")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()