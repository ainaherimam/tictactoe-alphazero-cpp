import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

"""
JAX Training Script for AlphaZero
==================================
Trains on data from /az_training shared memory segment.
Event-driven: runs K gradient steps each time new data arrives.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
import signal
import sys

from src.training.training_shm_reader import TrainingShmReader
from src.models.alphazero_model import (
    TrainingConfig,
    create_train_state,
    train_step,
    save_checkpoint,
    load_checkpoint,
)


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

# Global shutdown flag for signal handling
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        print("\n[Training] 🛑 Shutdown requested (Ctrl+C) - finishing current step...")
        print("[Training] Press Ctrl+C again to force quit.")
    else:
        print("\n[Training] ⚠️  Force quit!")
        sys.exit(1)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(config: TrainingConfig):
    """Main training loop."""
    
    global _shutdown_requested
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("ALPHAZERO JAX TRAINING")
    print("="*70)
    print(f"\nConfig:")
    for key, value in vars(config).items():
        print(f"  {key:25} = {value}")
    print()
    print("💡 Press Ctrl+C once to save and exit gracefully")
    print("   Press Ctrl+C twice to force quit\n")
    
    # Initialize JAX
    rng = jax.random.PRNGKey(42)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Connect to training data
    try:
        reader = TrainingShmReader(segment_name="/az_training")
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted during initialization (Ctrl+C)")
        print("[Training] Exiting...\n")
        return
    except Exception as e:
        print(f"\n[Training] ❌ Failed to connect to shared memory: {e}")
        print("[Training] Make sure the C++ self-play process is running first.\n")
        return
    
    # Wait for initial data
    try:
        reader.wait_for_data(min_positions=config.min_positions)
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted while waiting for data (Ctrl+C)")
        reader.close()
        print("[Training] Exiting...\n")
        return
    
    # Initialize model
    print("[Training] Initializing model...")
    state = create_train_state(rng, config)
    
    print(f"[Training] Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,} parameters\n")
    
    # Training loop
    print("[Training] Starting training loop...\n")
    print("-"*70)
    
    last_gen = reader.generation
    step = 0
    start_time = time.time()
    saved_checkpoints = set()  # Track which generations we've saved

    checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_0"
    if not checkpoint_path.exists():
        save_checkpoint(state, config.checkpoint_dir, step=0)
        saved_checkpoints.add(0)
    
    rng_np = np.random.default_rng(42)
    
    try:
        while not reader.is_shutdown() and not _shutdown_requested:
            try:
                # Poll for new data
                current_gen = reader.generation
                
                if current_gen == last_gen:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                last_gen = current_gen
                
                # Skip training if not at a 100-generation milestone
                if current_gen % 100 != 0:
                    continue
                
                for _ in range(config.steps_per_generation):
                    if _shutdown_requested:
                        break
                        
                    step += 1
                    
                    # Sample batch
                    batch_dict = reader.sample_batch(config.batch_size, rng=rng_np)
                    
                    # DIAGNOSTIC: Print data statistics on first few steps
                    if step <= 3:
                        print(f"\n[DIAGNOSTIC Step {step}]")
                        print(f"  z values - unique: {np.unique(batch_dict['z'])}, mean: {np.mean(batch_dict['z']):.4f}, std: {np.std(batch_dict['z']):.4f}")
                        print(f"  pi - sample row 0: {batch_dict['pi'][0][:8]}... (sum: {np.sum(batch_dict['pi'][0]):.4f})")
                        print(f"  pi - all rows same? {np.all(batch_dict['pi'] == batch_dict['pi'][0])}")
                        print(f"  pi - unique rows in first 50: {len(set(tuple(row) for row in batch_dict['pi'][:50]))}")
                        print(f"  boards - unique values: {len(np.unique(batch_dict['boards']))}")
                        print(f"  boards - all same? {np.all(batch_dict['boards'] == batch_dict['boards'][0])}")
                    
                    # Convert to JAX arrays
                    batch = {
                        'boards': jnp.array(batch_dict['boards']),
                        'pi': jnp.array(batch_dict['pi']),
                        'z': jnp.array(batch_dict['z']),
                        'mask': jnp.array(batch_dict['mask']),
                    }
                    
                    # Gradient step
                    state, metrics = train_step(state, batch)
                    
                    # Logging
                    if step % config.log_every_n_steps == 0:
                        elapsed = time.time() - start_time
                        samples_per_sec = (step * config.batch_size) / max(elapsed, 1.0)
                        
                        if config.verbose:
                            print(f"Step {step:6d} | Gen {current_gen:5d} | "
                                  f"Loss: {metrics['loss']:.4f} | "
                                  f"π: {metrics['policy_loss']:.4f} | "
                                  f"v: {metrics['value_loss']:.4f} | "
                                  f"H: {metrics['policy_entropy']:.3f} | "
                                  f"Acc: {metrics['value_accuracy']:.3f} | "
                                  f"Grad: {metrics['grad_norm']:.3f} | "
                                  f"{samples_per_sec:.0f} samples/s")
                        else:
                            print(f"Step {step:6d} | Loss: {metrics['loss']:.4f}")
                
                # Checkpoint saving (every 1000 generations)
                if current_gen % 300 == 0 and current_gen not in saved_checkpoints:
                    checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{current_gen}"
                    if not checkpoint_path.exists():
                        save_checkpoint(state, config.checkpoint_dir, step=current_gen)
                        saved_checkpoints.add(current_gen)
                        
                        # Create sentinel file to trigger reload and evaluations
                        checkpoint_dir = Path(config.checkpoint_dir)
                        sentinels = [
                            f"reload.mcts_jax_inference.{current_gen}.trigger",
                            f"reload.mcts_candidate_model.{current_gen}.trigger",
                            f"run.evaluation.{current_gen}.trigger",
                        ]

                        # Create all sentinels atomically
                        for sentinel_name in sentinels:
                            sentinel = checkpoint_dir / sentinel_name
                            sentinel.touch(exist_ok=True)
                                                
                        print(f"[Training] 💾 Saved checkpoint at generation {current_gen}")
                        print(f"[Training] 🔔 Created reload trigger")
                    else:
                        saved_checkpoints.add(current_gen)
                        print(f"[Training] ⏭️  Skipping checkpoint {current_gen} (already exists)")
            
            except KeyboardInterrupt:
                # This shouldn't happen with signal handler, but just in case
                print("\n[Training] 🛑 Interrupted by user (Ctrl+C)")
                print("[Training] Saving checkpoint before exit...")
                break
        
        # Check why we exited
        if _shutdown_requested:
            print("\n[Training] Graceful shutdown completed")
        elif reader.is_shutdown():
            print("\n[Training] Self-play process signaled shutdown")
    
    except Exception as e:
        print(f"\n[Training] ❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='JAX AlphaZero Training')
    
    # Data
    parser.add_argument('--min-positions', type=int, default=2000,
                       help='Wait for this many positions before starting')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--steps-per-gen', type=int, default=5,
                       help='Gradient steps per generation')
    
    # Model
    parser.add_argument('--channels', type=int, default=64,
                       help='Residual block channels')
    parser.add_argument('--blocks', type=int, default=3,
                       help='Number of residual blocks')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=1,
                       help='Log every N steps')
    parser.add_argument('--quiet', action='store_true',
                       help='Less verbose logging')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.min_positions = args.min_positions
    config.batch_size = args.batch_size
    config.steps_per_generation = args.steps_per_gen
    config.num_channels = args.channels
    config.num_res_blocks = args.blocks
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.log_every_n_steps = args.log_every
    config.verbose = not args.quiet
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()