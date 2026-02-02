"""
JAX Training Script for AlphaZero
==================================
Trains on data from /az_training shared memory segment.
Event-driven: runs K gradient steps each time new data arrives.
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Dict, Tuple, Any
import numpy as np
import time
from pathlib import Path
import pickle
import signal
import sys

from training_shm_reader import TrainingShmReader


# ============================================================================
# CONFIGURATION
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


class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Data settings
    min_positions: int = 1024           # Wait for this many before starting
    batch_size: int = 32               # Positions per gradient step
    steps_per_generation: int = 10      # Steps to run when new data arrives
    
    # Model architecture
    num_channels: int = 64              # Residual block channels
    num_res_blocks: int = 5             # Number of residual blocks
    
    # Optimization
    learning_rate: float = 0.001        # Initial learning rate
    weight_decay: float = 1e-4          # L2 regularization
    grad_clip: float = 1.0              # Gradient clipping threshold
    
    # Learning rate schedule
    lr_schedule: str = "cosine"         # "constant" or "cosine"
    lr_warmup_steps: int = 100          # Warmup steps for cosine schedule
    lr_min: float = 1e-5                # Minimum LR for cosine schedule
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_gens: int = 50        # Save checkpoint every N generations
    
    # Logging
    log_every_n_steps: int = 10         # Print metrics every N steps
    verbose: bool = True                # Detailed logging


# ============================================================================
# MODEL DEFINITION (JAX/Flax)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    num_channels: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = x + residual
        x = nn.relu(x)
        
        return x


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network: board → (policy, value)."""
    num_channels: int = 64
    num_res_blocks: int = 5
    num_actions: int = 16
    
    @nn.compact
    def __call__(self, x, mask, training: bool = True):
        # Input conv
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Residual tower
        for _ in range(self.num_res_blocks):
            x = ResidualBlock(self.num_channels)(x, training=training)
        
        # Policy head
        p = nn.Conv(2, (1, 1))(x)
        p = nn.BatchNorm(use_running_average=not training)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))  # Flatten
        p = nn.Dense(self.num_actions)(p)
        
        # Apply mask and log-softmax
        p = jnp.where(mask > 0, p, -1e9)
        p = jax.nn.log_softmax(p, axis=-1)
        
        # Value head
        v = nn.Conv(1, (1, 1))(x)
        v = nn.BatchNorm(use_running_average=not training)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))  # Flatten
        v = nn.Dense(64)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v).squeeze(-1)
        
        return p, v


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def alphazero_loss(params, state, batch):
    """
    Compute AlphaZero loss: cross-entropy(policy) + MSE(value).
    
    Args:
        params: Model parameters
        state: Training state
        batch: Dict with keys 'boards', 'pi', 'z', 'mask'
    
    Returns:
        (total_loss, metrics_dict)
    """
    boards = batch['boards']  # [B, 3, 4, 4]
    pi_target = batch['pi']   # [B, 16]
    z_target = batch['z']     # [B]
    mask = batch['mask']      # [B, 16]
    
    # Forward pass
    (p_pred, v_pred), updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        boards, 
        mask,
        training=True,
        mutable=['batch_stats']
    )
    
    # Policy loss: cross-entropy with target distribution
    policy_loss = -jnp.sum(pi_target * p_pred, axis=-1).mean()
    
    # Value loss: MSE
    value_loss = jnp.mean((v_pred - z_target) ** 2)
    
    # Total loss
    total_loss = policy_loss + value_loss
    
    # Metrics
    metrics = {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'policy_entropy': -jnp.sum(jnp.exp(p_pred) * p_pred, axis=-1).mean(),
        'value_accuracy': jnp.mean(jnp.abs(v_pred - z_target) < 0.4),
    }
    
    return total_loss, (metrics, updates['batch_stats'])


# ============================================================================
# TRAINING STATE
# ============================================================================

def create_train_state(rng, config: TrainingConfig):
    """Initialize model and optimizer."""
    
    # Create model
    model = AlphaZeroNet(
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks,
        num_actions=16
    )
    
    # Initialize with dummy input
    dummy_board = jnp.zeros((1, 3, 4, 4))
    dummy_mask = jnp.ones((1, 16))
    
    variables = model.init(rng, dummy_board, dummy_mask, training=False)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Create optimizer with weight decay and gradient clipping
    if config.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=10000,  # Will be updated dynamically
            end_value=config.lr_min,
        )
    else:
        schedule = config.learning_rate
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    # Create training state with batch_stats
    class TrainStateWithBatchStats(train_state.TrainState):
        batch_stats: Any
    
    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    
    return state


@jax.jit
def train_step(state, batch):
    """Single gradient step."""
    
    grad_fn = jax.value_and_grad(alphazero_loss, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(state.params, state, batch)
    
    # Update parameters and batch stats
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    
    # Add gradient norm to metrics
    metrics['grad_norm'] = optax.global_norm(grads)
    
    return state, metrics


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
    
    rng_np = np.random.default_rng(42)
    
    try:
        while not reader.is_shutdown() and not _shutdown_requested:
            try:
                # Poll for new data
                current_gen = reader.generation
                
                if current_gen == last_gen:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # New data arrived! Run K gradient steps
                last_gen = current_gen
                
                for _ in range(config.steps_per_generation):
                    if _shutdown_requested:
                        break
                        
                    step += 1
                    
                    # Sample batch
                    batch_dict = reader.sample_batch(config.batch_size, rng=rng_np)
                    
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
                
                # Checkpoint saving
                if current_gen % config.save_every_n_gens == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_gen{current_gen}.pkl"
                    save_checkpoint(state, checkpoint_path)
                    print(f"[Training] 💾 Saved checkpoint: {checkpoint_path}")
            
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
    
    finally:
        # Final checkpoint
        try:
            final_path = checkpoint_dir / "checkpoint_final.pkl"
            save_checkpoint(state, final_path)
            print(f"[Training] 💾 Saved final checkpoint: {final_path}")
        except Exception as e:
            print(f"[Training] ⚠️  Warning: Could not save final checkpoint: {e}")
        
        # Close reader
        try:
            reader.close()
            print("[Training] ✅ Training finished\n")
        except Exception as e:
            print(f"[Training] ⚠️  Warning: Error closing reader: {e}\n")


def save_checkpoint(state, path):
    """Save training state to disk."""
    checkpoint = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'step': state.step,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(state, path):
    """Load training state from disk."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    state = state.replace(
        params=checkpoint['params'],
        batch_stats=checkpoint['batch_stats'],
        step=checkpoint['step'],
    )
    return state


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
    parser.add_argument('--steps-per-gen', type=int, default=10,
                       help='Gradient steps per generation')
    
    # Model
    parser.add_argument('--channels', type=int, default=64,
                       help='Residual block channels')
    parser.add_argument('--blocks', type=int, default=5,
                       help='Number of residual blocks')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=10,
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