"""
AlphaZero Neural Network Model - JAX/Flax Implementation
=========================================================
Contains all model-related components:
- Network architecture (ResNet-style)
- Loss function
- Training state management
- JIT-compiled training step
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Any, Dict, Optional, TYPE_CHECKING
import orbax.checkpoint as ocp
from pathlib import Path

from src.constants import INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, POLICY_SIZE

if TYPE_CHECKING:
    from src.core.solver.misere_solver import MisereSolver


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
    """AlphaZero neural network: board → (policy, value).

    Value head:
      - categorical=False (default): scalar tanh in [-1, 1]  — trained with MSE
      - categorical=True:  log-probabilities over 3 bins {-1, 0, +1}  — trained with CE
    """
    num_channels: int = 64
    num_res_blocks: int = 5
    num_actions: int = 16
    categorical: bool = False

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

        # Apply log-softmax
        p = jax.nn.log_softmax(p, axis=-1)

        # Value head
        v = nn.Conv(1, (1, 1))(x)
        v = nn.BatchNorm(use_running_average=not training)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))  # Flatten
        v = nn.Dense(64)(v)
        v = nn.relu(v)
        if self.categorical:
            v = nn.Dense(3)(v)                      # logits for bins [-1, 0, +1]
            v = jax.nn.log_softmax(v, axis=-1)      # [B, 3] log-probabilities
        else:
            v = nn.Dense(1)(v)                      # [B, 1]
            v = jnp.tanh(v).squeeze(-1)             # [B] scalar in [-1, 1]

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


def scalar_to_categorical(z):
    """Map z ∈ {-1, 0, +1} to one-hot over bins [loss, draw, win]."""
    bin_idx = (z + 1).astype(jnp.int32)              # {-1,0,+1} → {0,1,2}
    return jax.nn.one_hot(bin_idx, num_classes=3)     # [B, 3]


def alphazero_loss(params, state, batch, train_value_only: bool = False,
                   categorical: bool = False):
    """
    Compute AlphaZero loss: cross-entropy(policy) + value loss.

    Value loss is MSE for scalar head, cross-entropy for categorical head.

    When train_value_only=True, only value_loss contributes to the gradient.
    Policy head parameters receive zero gradient and remain at their initial
    (random) values, while the shared body and value head are trained normally.

    Args:
        params: Model parameters
        state: Training state
        batch: Dict with keys 'boards', 'pi', 'z', 'mask'
        train_value_only: If True, exclude policy loss from the gradient
        categorical: If True, use categorical CE loss on 3-bin value head
    Returns:
        (total_loss, (metrics_dict, new_batch_stats))
    """
    boards = batch['boards']  # [B, 2, 4, 4]
    pi_target = batch['pi']   # [B, 16]
    z_target = batch['z']     # [B]  values in {-1, 0, +1}
    mask = batch['mask']      # [B, 16]

    # Forward pass
    (p_pred, v_pred), updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        boards,
        mask,
        training=True,
        mutable=['batch_stats']
    )
    # p_pred: [B, 16] log-probs
    # v_pred: [B] scalar tanh  OR  [B, 3] log-probs (categorical)

    # Policy loss: cross-entropy with target distribution
    policy_loss = -jnp.sum(pi_target * p_pred, axis=-1).mean()

    # Value loss
    if categorical:
        z_onehot = scalar_to_categorical(z_target)              # [B, 3]
        value_loss = -jnp.sum(z_onehot * v_pred, axis=-1).mean()
    else:
        value_loss = jnp.mean((v_pred - z_target.astype(jnp.float32)) ** 2)

    # Total loss — when train_value_only, drop policy gradient entirely.
    # policy_loss is still computed above for metric reporting.
    total_loss = value_loss if train_value_only else policy_loss + value_loss

    # Policy accuracy metrics
    top1_target = jnp.argmax(pi_target, axis=-1)
    top1_pred   = jnp.argmax(p_pred,   axis=-1)

    # Top-k indices of predicted logits (ascending, so slice from the end)
    top3_pred_indices = jnp.argsort(p_pred, axis=-1)[:, -3:]
    top2_pred_indices = top3_pred_indices[:, -2:]

    # Check whether the greedy target action appears within top-k predictions
    policy_top1_acc = jnp.mean(top1_pred == top1_target)

    policy_top2_acc = jnp.mean(
        jnp.any(top2_pred_indices == top1_target[:, None], axis=-1)
    )

    policy_top3_acc = jnp.mean(
        jnp.any(top3_pred_indices == top1_target[:, None], axis=-1)
    )

    # Value accuracy
    if categorical:
        # Predicted bin matches true bin
        value_accuracy = jnp.mean(
            jnp.argmax(v_pred, axis=-1) == (z_target + 1).astype(jnp.int32)
        )
    else:
        # Predicted sign matches true sign (win/loss/draw)
        value_accuracy = jnp.mean(jnp.sign(v_pred) == jnp.sign(z_target))

    # Metrics
    metrics = {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'policy_entropy': -jnp.sum(jnp.exp(p_pred) * p_pred, axis=-1).mean(),
        'value_accuracy': value_accuracy,
        'policy_top1_acc': policy_top1_acc,   # pred top-1 == target top-1
        'policy_top2_acc': policy_top2_acc,   # target top-1 in pred top-2
        'policy_top3_acc': policy_top3_acc,   # target top-1 in pred top-3
    }

    return total_loss, (metrics, updates['batch_stats'])

# ============================================================================
# TRAINING STEP (JIT-COMPILED)
# ============================================================================

@functools.partial(jax.jit, static_argnums=(2, 3))
def train_step(state, batch, train_value_only: bool = False,
               categorical: bool = False):
    """
    Single gradient step.

    Args:
        state: Training state with params, optimizer state, batch_stats
        batch: Dictionary with 'boards', 'pi', 'z', 'mask'
        train_value_only: If True, only the value head (and shared body) are
            trained; policy head parameters receive zero gradient and remain
            at their initial random values.
        categorical: If True, use categorical value head + CE loss.

    Returns:
        (updated_state, metrics_dict)
    """
    grad_fn = jax.value_and_grad(alphazero_loss, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(
        state.params, state, batch, train_value_only, categorical
    )
    
    # Update parameters and batch stats
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    
    # Add gradient norm to metrics
    metrics['grad_norm'] = optax.global_norm(grads)
    
    return state, metrics


# ============================================================================
# SOLVER-BASED POLICY ACCURACY METRICS
# ============================================================================

def compute_solver_metrics(
    boards: np.ndarray,
    pi_mcts: np.ndarray,
    p_pred_log: np.ndarray,
    solver: "MisereSolver",
) -> Dict[str, float]:
    """
    Measure how well MCTS targets and NN predictions agree with the perfect solver.

    For each board position the solver returns all optimal moves (moves that
    achieve the game-theoretic value).  We then check whether the greedy
    action from each policy falls among those optimal moves.

    Args:
        boards:      [B, 2, 4, 4]  float32 board planes (current, opponent)
        pi_mcts:     [B, 16]       MCTS visit-count policy (the training target π)
        p_pred_log:  [B, 16]       NN log-probabilities (output of the network)
        solver:      pre-solved MisereSolver instance (call solver.solve() once first)

    Returns:
        Dict with:
          'policy_acc_mcts' — fraction of positions where MCTS top-1 is solver-optimal
          'policy_acc_nn'   — fraction of positions where NN  top-1 is solver-optimal
    """
    from src.core.solver.misere_solver import board_to_masks

    boards_np   = np.asarray(boards)
    pi_np       = np.asarray(pi_mcts)
    p_pred_np   = np.asarray(p_pred_log)

    B = boards_np.shape[0]
    mcts_correct = 0
    nn_correct   = 0
    valid_count  = 0

    for i in range(B):
        bx, bo, is_x_turn = board_to_masks(boards_np[i])
        optimal = set(solver.get_optimal_moves(bx, bo, is_x_turn))
        if not optimal:
            continue

        valid_count += 1

        if int(np.argmax(pi_np[i])) in optimal:
            mcts_correct += 1

        # Mask occupied cells before argmax: NN log-softmax is over all 16
        # cells, so the raw argmax can land on an occupied cell.
        occupied = bx | bo
        nn_logits = p_pred_np[i].copy()
        for c in range(16):
            if occupied & (1 << c):
                nn_logits[c] = -np.inf
        if int(np.argmax(nn_logits)) in optimal:
            nn_correct += 1

    if valid_count == 0:
        return {"policy_acc_mcts": 0.0, "policy_acc_nn": 0.0}

    return {
        "policy_acc_mcts": mcts_correct / valid_count,
        "policy_acc_nn":   nn_correct   / valid_count,
    }


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
    evaluate_every_n_gens: int = 300
    train_every_n_gens: int = 100
    
    # Training mode
    train_value_only: bool = False       # If True, freeze policy head (value head + body only)
    categorical: bool = False            # If True, use categorical 3-bin value head + CE loss

    # Logging
    log_every_n_steps: int = 10         # Print metrics every N steps
    verbose: bool = True                # Detailed logging


def create_inference_state(rng, num_channels: int = 64, num_res_blocks: int = 3,
                           num_actions: int = POLICY_SIZE, categorical: bool = False):
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
        num_actions=num_actions,
        categorical=categorical,
    )

    # Initialize with dummy input
    dummy_board = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
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
        num_actions=POLICY_SIZE,
        categorical=config.categorical,
    )

    # Initialize with dummy input
    dummy_board = jnp.zeros((1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    dummy_mask = jnp.ones((1, POLICY_SIZE))
    
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
    num_actions: int = 16,
    categorical: bool = False,
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
        num_actions=num_actions,
        categorical=categorical,
    )

    print(f"[Checkpoint] ✓ Loaded inference weights")
    
    return {
        'params': restored['params'],
        'batch_stats': restored.get('batch_stats', {}),
        'apply_fn': model.apply,
    }