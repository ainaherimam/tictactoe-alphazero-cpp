"""
AlphaZero Neural Network Model - JAX/Flax Implementation
=========================================================
Contains all model-related components:
- Network architecture (ResNet-style)
- Loss function
- Training state management
- JIT-compiled training step
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Any, Dict, Optional
import orbax.checkpoint as ocp
from pathlib import Path


# ============================================================================
# MODEL ARCHITECTURE
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
# TRAINING STATE
# ============================================================================

class TrainStateWithBatchStats(train_state.TrainState):
    """Extended training state with batch statistics for batch norm."""
    batch_stats: Any


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
        (total_loss, (metrics_dict, new_batch_stats))
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
# TRAINING STEP (JIT-COMPILED)
# ============================================================================

@jax.jit
def train_step(state, batch):
    """
    Single gradient step.
    
    Args:
        state: Training state with params, optimizer state, batch_stats
        batch: Dictionary with 'boards', 'pi', 'z', 'mask'
    
    Returns:
        (updated_state, metrics_dict)
    """
    grad_fn = jax.value_and_grad(alphazero_loss, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(state.params, state, batch)
    
    # Update parameters and batch stats
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    
    # Add gradient norm to metrics
    metrics['grad_norm'] = optax.global_norm(grads)
    
    return state, metrics


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Data settings
    min_positions: int = 1024           # Wait for this many before starting
    batch_size: int = 128               # Positions per gradient step
    steps_per_generation: int = 10      # Steps to run when new data arrives
    
    # Model architecture
    num_channels: int = 64              # Residual block channels
    num_res_blocks: int = 3             # Number of residual blocks
    
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
    save_every_n_gens: int = 10        # Save checkpoint every N generations
    
    # Logging
    log_every_n_steps: int = 10         # Print metrics every N steps
    verbose: bool = True                # Detailed logging


def create_inference_state(rng, num_channels: int = 64, num_res_blocks: int = 3, num_actions: int = 16):
    """
    Initialize model for inference only (no optimizer).
    
    Args:
        rng: JAX random key
        num_channels: Number of channels in residual blocks
        num_res_blocks: Number of residual blocks
        num_actions: Number of possible actions (policy size)
    
    Returns:
        Dictionary with 'params', 'batch_stats', and 'apply_fn'
    """
    # Create model
    model = AlphaZeroNet(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_actions=num_actions
    )
    
    # Initialize with dummy input
    dummy_board = jnp.zeros((1, 3, 4, 4))
    dummy_mask = jnp.ones((1, num_actions))
    
    variables = model.init(rng, dummy_board, dummy_mask, training=False)
    
    return {
        'params': variables['params'],
        'batch_stats': variables['batch_stats'],
        'apply_fn': model.apply,
    }


def create_train_state(rng, config: TrainingConfig):
    """
    Initialize model and optimizer.
    
    Args:
        rng: JAX random key
        config: Training configuration
    
    Returns:
        Initialized training state with params, optimizer, and batch_stats
    """
    
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
    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    
    return state


# ============================================================================
# CHECKPOINTING (ORBAX)
# ============================================================================

# Global checkpointer instance (reused across saves/loads)
_checkpointer = None

def get_checkpointer():
    """Get or create the global checkpointer instance."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = ocp.StandardCheckpointer()
    return _checkpointer


def save_checkpoint(state, checkpoint_dir: str, step: Optional[int] = None):
    """
    Save model weights using Orbax (params and batch_stats only).
    
    Args:
        state: Training state
        checkpoint_dir: Directory to save checkpoints
        step: Training step number (uses state.step if None)
    """

    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint path
    step = int(step if step is not None else state.step)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step}"

    # Only save weights (params and batch_stats)
    checkpoint_data = {
        'params': state.params,
        'batch_stats': state.batch_stats,
    }

    checkpointer = get_checkpointer()
    checkpointer.save(checkpoint_path, checkpoint_data)

    print(f"[Checkpoint] ✓ Saved weights to checkpoint_{step}")



def load_checkpoint(state, checkpoint_path: str, learning_rate: float = 1e-3, 
                    weight_decay: float = 1e-4, grad_clip: float = 1.0) -> 'TrainStateWithBatchStats':
    """
    Load model weights from Orbax checkpoint and create fresh optimizer.
    
    Args:
        state: Template training state (for structure)
        checkpoint_path: Path to checkpoint directory
        learning_rate: Learning rate for new optimizer
        weight_decay: Weight decay for new optimizer
        grad_clip: Gradient clipping threshold for new optimizer
    
    Returns:
        New training state with loaded weights and fresh optimizer
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"[Checkpoint] Loading weights from {checkpoint_path}")
    
    # Load checkpoint
    checkpointer = get_checkpointer()
    restored = checkpointer.restore(checkpoint_path)
    
    # Create fresh optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    
    # Create new training state with loaded weights
    state = TrainStateWithBatchStats.create(
        apply_fn=state.apply_fn,
        params=restored['params'],
        tx=tx,
        batch_stats=restored.get('batch_stats', {}),
    )
    
    print(f"[Checkpoint] ✓ Loaded weights (created fresh optimizer)")
    return state


def load_checkpoint_for_inference(
    checkpoint_path: str,
    num_channels: int = 64,
    num_res_blocks: int = 3,
    num_actions: int = 16
) -> Dict:
    """
    Load checkpoint for inference only (no optimizer).
    
    This loads only the model weights (params and batch_stats).
    
    Args:
        checkpoint_path: Path to checkpoint directory
        num_channels: Number of channels in model
        num_res_blocks: Number of residual blocks  
        num_actions: Number of possible actions
    
    Returns:
        Dictionary with 'params', 'batch_stats', 'apply_fn'
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"[Checkpoint] Loading inference weights from {checkpoint_path}")
    
    # Load checkpoint directly with Orbax
    checkpointer = get_checkpointer()
    restored = checkpointer.restore(checkpoint_path)
    
    # Create model for apply_fn
    model = AlphaZeroNet(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_actions=num_actions
    )
    
    print(f"[Checkpoint] ✓ Loaded inference weights")
    
    return {
        'params': restored['params'],
        'batch_stats': restored.get('batch_stats', {}),
        'apply_fn': model.apply,
    }