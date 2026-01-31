#!/usr/bin/env python3
"""
AlphaZero Neural Network Model with JAX/Flax and Orbax Checkpointing
IMPROVED VERSION with JIT compilation and fixed learning rate scheduling
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import time
from functools import partial


# ============================================================================
# Model Architecture
# ============================================================================

class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style neural network with policy and value heads
    
    Architecture:
    - Shared convolutional representation
    - Residual blocks
    - Separate policy and value heads
    """
    num_actions: int
    num_channels: int = 64
    num_res_blocks: int = 3
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, channels, height, width]
            training: Whether in training mode (for batch norm)
            
        Returns:
            (policy_logits, value): Policy logits and value estimate
        """
        # Initial convolutional block
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Residual blocks
        for _ in range(self.num_res_blocks):
            residual = x
            x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = x + residual
            x = nn.relu(x)
        
        # Policy head
        p = nn.Conv(features=2, kernel_size=(1, 1))(x)
        p = nn.BatchNorm(use_running_average=not training)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))
        policy_logits = nn.Dense(features=self.num_actions)(p)
        
        # Value head
        v = nn.Conv(features=1, kernel_size=(1, 1))(x)
        v = nn.BatchNorm(use_running_average=not training)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(features=64)(v)
        v = nn.relu(v)
        value = nn.Dense(features=1)(v)
        value = jnp.tanh(value).squeeze(-1)
        
        return policy_logits, value


# ============================================================================
# Training State
# ============================================================================

class TrainState(train_state.TrainState):
    """Extended training state with batch statistics for batch norm"""
    batch_stats: Any = None


# ============================================================================
# Model Manager
# ============================================================================

class AlphaZeroModel:
    """
    Manages AlphaZero model lifecycle: initialization, loading, training, and checkpointing
    """
    
    def __init__(self, 
                 num_actions: int,
                 input_channels: int = 3,
                 board_height: int = 4,
                 board_width: int = 4,
                 num_channels: int = 64,
                 num_res_blocks: int = 3):
        """
        Initialize model configuration
        
        Args:
            num_actions: Number of possible actions (policy output size)
            input_channels: Number of input channels
            board_height: Board height
            board_width: Board width
            num_channels: Number of channels in conv layers
            num_res_blocks: Number of residual blocks
        """
        self.num_actions = num_actions
        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        
        # Create model
        self.model = AlphaZeroNet(
            num_actions=num_actions,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks
        )
        
        self.state = None
        self.checkpointer = None
        
        # JIT-compiled inference function (created on first call)
        self._inference_fn = None
        
    def initialize(self, 
                   learning_rate: float = 1e-3,
                   weight_decay: float = 1e-4,
                   seed: int = 0) -> TrainState:
        """
        Initialize model parameters and optimizer
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            seed: Random seed
            
        Returns:
            Initialized training state
        """
        print(f"[Model] Initializing with lr={learning_rate}, weight_decay={weight_decay}")
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(seed)
        init_rng, dropout_rng = jax.random.split(rng)
        
        dummy_input = jnp.zeros((1, self.input_channels, self.board_height, self.board_width))
        variables = self.model.init(init_rng, dummy_input, training=True)
        
        # Extract params and batch_stats
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})
        
        # Create optimizer with weight decay (AdamW)
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            batch_stats=batch_stats
        )
        
        # Create JIT-compiled inference function
        self._create_inference_fn()
        
        print(f"[Model] ✓ Initialized with {self._count_parameters()} parameters")
        return self.state
    
    def _create_inference_fn(self):
        """Create JIT-compiled inference function"""
        @jax.jit
        def inference(params, batch_stats, x):
            """Fast inference function"""
            variables = {'params': params, 'batch_stats': batch_stats}
            policy_logits, value = self.model.apply(
                variables, x, training=False, mutable=False
            )
            return policy_logits, value
        
        self._inference_fn = inference
    
    def predict(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fast inference using JIT-compiled function
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            (policy_logits, value): Policy logits and value estimates
        """
        if self.state is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if self._inference_fn is None:
            self._create_inference_fn()
        
        return self._inference_fn(self.state.params, self.state.batch_stats, x)
    
    def load(self, checkpoint_path: str) -> TrainState:
        """
        Load model from Orbax checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint directory or file
            
        Returns:
            Loaded training state
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        print(f"[Model] Loading checkpoint from {checkpoint_path}")
        
        # Create checkpointer if not exists
        if self.checkpointer is None:
            self.checkpointer = ocp.StandardCheckpointer()
        
        # Load checkpoint
        restored = self.checkpointer.restore(checkpoint_path)
        
        # Reconstruct training state
        # Initialize optimizer (we'll restore the optimizer state from checkpoint)
        optimizer = optax.adamw(
            learning_rate=restored.get('learning_rate', 1e-3),
            weight_decay=restored.get('weight_decay', 1e-4)
        )
        
        self.state = TrainState(
            step=restored['step'],
            apply_fn=self.model.apply,
            params=restored['params'],
            tx=optimizer,
            opt_state=restored['opt_state'],
            batch_stats=restored.get('batch_stats', {})
        )
        
        # Recreate JIT inference function
        self._create_inference_fn()
        
        print(f"[Model] ✓ Loaded checkpoint at step {self.state.step}")
        return self.state
    
    def save(self, checkpoint_dir: str, step: Optional[int] = None):
        """
        Save model checkpoint using Orbax
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            step: Training step number (optional)
        """
        if self.state is None:
            raise RuntimeError("No model state to save. Initialize or load model first.")
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpointer if not exists
        if self.checkpointer is None:
            self.checkpointer = ocp.StandardCheckpointer()
        
        # Determine checkpoint path
        if step is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{self.state.step}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': int(self.state.step),
            'params': self.state.params,
            'opt_state': self.state.opt_state,
            'batch_stats': self.state.batch_stats,
        }
        
        # Save checkpoint
        self.checkpointer.save(checkpoint_path, checkpoint_data)
        print(f"[Model] ✓ Saved checkpoint to {checkpoint_path}")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        if self.state is None:
            return 0
        return sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))


# ============================================================================
# Loss and Training
# ============================================================================

def compute_loss(params, batch_stats, apply_fn, inputs, targets, masks):
    """
    Compute combined policy and value loss
    
    Args:
        params: Model parameters
        batch_stats: Batch normalization statistics
        apply_fn: Model forward function
        inputs: Input batch [batch, channels, height, width]
        targets: Target batch [batch, num_actions + 1] (policy + value)
        masks: Legal action masks [batch, num_actions]
        
    Returns:
        (loss, (new_batch_stats, metrics))
    """
    # Forward pass with batch norm updates
    variables = {'params': params, 'batch_stats': batch_stats}
    (policy_logits, value), new_vars = apply_fn(
        variables, inputs, training=True, mutable=['batch_stats']
    )
    new_batch_stats = new_vars['batch_stats']
    
    # Extract targets
    policy_target = targets[:, :-1]
    value_target = targets[:, -1]
    
    # Policy loss (cross entropy with masking)
    masked_logits = jnp.where(masks > 0, policy_logits, -1e9)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    policy_loss = -jnp.mean(jnp.sum(policy_target * log_probs, axis=-1))
    
    # Value loss (MSE)
    value_loss = jnp.mean(jnp.square(value - value_target))
    
    # Combined loss
    total_loss = policy_loss + value_loss
    
    # Compute metrics
    policy_probs = jax.nn.softmax(masked_logits, axis=-1)
    policy_entropy = -jnp.mean(jnp.sum(policy_probs * log_probs, axis=-1))
    
    # Value accuracy (within 0.1)
    value_accuracy = jnp.mean(jnp.abs(value - value_target) < 0.1)
    
    # Sign accuracy (same sign)
    sign_accuracy = jnp.mean((value * value_target) > 0)
    
    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
        'policy_entropy': policy_entropy,
        'value_accuracy': value_accuracy,
        'sign_accuracy': sign_accuracy,
    }
    
    return total_loss, (new_batch_stats, metrics)


@jax.jit
def train_step(state: TrainState, inputs, targets, masks):
    """
    Single training step
    
    Args:
        state: Training state
        inputs: Input batch
        targets: Target batch
        masks: Legal move masks
        
    Returns:
        (new_state, metrics)
    """
    def loss_fn(params):
        return compute_loss(
            params, state.batch_stats, state.apply_fn,
            inputs, targets, masks
        )
    
    # Compute gradients
    (loss, (new_batch_stats, metrics)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    
    # Compute gradient norm
    grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)
    ))
    metrics['gradient_norm'] = grad_norm
    
    # Update parameters
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=new_batch_stats
    )
    
    return new_state, metrics


# ============================================================================
# Training Function (FIXED)
# ============================================================================

def train(
    model: AlphaZeroModel,
    dataset,  # Should have .get(idx) method returning (data, target)
    batch_size: int,
    training_steps: int,
    learning_rate: float = 1e-3,
    min_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 1000,
    log_interval: int = 100,
    logger=None,  # Optional logger with add_scalar and flush_metrics methods
    current_iteration: int = 0,
    global_step: int = 0,
    seed: int = 0
):
    """
    Train the AlphaZero model with FIXED learning rate scheduling
    
    Key fixes:
    - Use optax.inject_hyperparams to update learning rate without resetting optimizer state
    - Proper cosine annealing schedule
    """
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {training_steps}")
    print(f"Learning rate: {learning_rate} -> {min_lr}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*70}\n")
    
    # Initialize model if not already initialized
    if model.state is None:
        # Create optimizer with learning rate schedule
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=training_steps,
            alpha=min_lr / learning_rate
        )
        
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay
        )
        
        # Initialize with scheduled optimizer
        rng = jax.random.PRNGKey(seed)
        dummy_input = jnp.zeros((1, model.input_channels, model.board_height, model.board_width))
        variables = model.model.init(rng, dummy_input, training=True)
        
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})
        
        model.state = TrainState.create(
            apply_fn=model.model.apply,
            params=params,
            tx=optimizer,
            batch_stats=batch_stats
        )
        model._create_inference_fn()
    
    state = model.state
    
    # Random number generator for sampling
    rng = np.random.RandomState(seed)
    dataset_size = len(dataset)
    
    # Training statistics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    total_policy_entropy = 0.0
    total_value_accuracy = 0.0
    total_sign_accuracy = 0.0
    max_gradient_norm = 0.0
    
    start_time = time.time()
    
    for step in range(1, training_steps + 1):
        # Sample batch
        batch_indices = rng.randint(0, dataset_size, size=batch_size)
        batch_data = [dataset.get(idx) for idx in batch_indices]
        
        # Stack into batches
        inputs = jnp.stack([item[0] for item in batch_data])
        targets = jnp.stack([item[1] for item in batch_data])
        
        # Extract masks (assuming last columns after policy target)
        policy_size = model.num_actions
        masks = targets[:, policy_size + 1:]  # After value target
        targets = targets[:, :policy_size + 1]  # Policy + value only
        
        # Training step
        state, metrics = train_step(state, inputs, targets, masks)
        
        # Get current learning rate from optimizer state
        # (automatically handled by cosine_decay_schedule)
        current_lr = learning_rate  # Will be updated by schedule
        
        # Accumulate metrics
        total_policy_loss += float(metrics['policy_loss'])
        total_value_loss += float(metrics['value_loss'])
        total_loss += float(metrics['total_loss'])
        total_policy_entropy += float(metrics['policy_entropy'])
        total_value_accuracy += float(metrics['value_accuracy'])
        total_sign_accuracy += float(metrics['sign_accuracy'])
        max_gradient_norm = max(max_gradient_norm, float(metrics['gradient_norm']))
        
        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (step * batch_size) / max(elapsed, 1.0)
            
            avg_policy_loss = total_policy_loss / log_interval
            avg_value_loss = total_value_loss / log_interval
            avg_total_loss = total_loss / log_interval
            avg_policy_entropy = total_policy_entropy / log_interval
            avg_value_accuracy = total_value_accuracy / log_interval
            avg_sign_accuracy = total_sign_accuracy / log_interval
            
            # Calculate actual current LR
            progress = step / training_steps
            current_lr = min_lr + (learning_rate - min_lr) * \
                        (1 + np.cos(np.pi * progress)) / 2.0
            
            print(f"Step {step}/{training_steps} | "
                  f"Loss: {avg_total_loss:.4f} | "
                  f"Policy: {avg_policy_loss:.4f} | "
                  f"Value: {avg_value_loss:.4f} | "
                  f"Entropy: {avg_policy_entropy:.4f} | "
                  f"V-Acc: {avg_value_accuracy:.3f} | "
                  f"Sign-Acc: {avg_sign_accuracy:.3f} | "
                  f"LR: {current_lr:.2e} | "
                  f"GradNorm: {max_gradient_norm:.3f}")
            
            # Log to external logger if provided
            if logger is not None:
                current_global_step = global_step + step
                logger.add_scalar("global_step", current_global_step)
                logger.add_scalar("iteration", current_iteration)
                logger.add_scalar("training_step", step)
                logger.add_scalar("training/total_loss", float(metrics['total_loss']))
                logger.add_scalar("training/policy_loss", float(metrics['policy_loss']))
                logger.add_scalar("training/value_loss", float(metrics['value_loss']))
                logger.add_scalar("training/policy_entropy", float(metrics['policy_entropy']))
                logger.add_scalar("training/value_accuracy", float(metrics['value_accuracy']))
                logger.add_scalar("training/sign_accuracy", float(metrics['sign_accuracy']))
                logger.add_scalar("training/learning_rate", current_lr)
                logger.add_scalar("training/gradient_norm", float(metrics['gradient_norm']))
                logger.add_scalar("training/samples_per_sec", samples_per_sec)
                logger.flush_metrics()
            
            # Reset accumulators
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_loss = 0.0
            total_policy_entropy = 0.0
            total_value_accuracy = 0.0
            total_sign_accuracy = 0.0
            max_gradient_norm = 0.0
        
        # Checkpointing
        if checkpoint_dir and step % checkpoint_interval == 0:
            model.state = state
            model.save(checkpoint_dir, step=global_step + step)
    
    # Save final checkpoint
    if checkpoint_dir:
        model.state = state
        model.save(checkpoint_dir, step=global_step + training_steps)
    
    model.state = state
    print(f"\n✅ Training completed!\n")
    
    return state


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing AlphaZero Model with JIT inference...")
    
    # Initialize model
    model = AlphaZeroModel(
        num_actions=16,
        input_channels=3,
        board_height=4,
        board_width=4
    )
    
    # Initialize
    state = model.initialize(learning_rate=1e-3)
    
    # Test inference speed
    x = jnp.ones((1, 3, 4, 4))
    
    # Warm-up (JIT compilation)
    print("\nWarming up JIT compilation...")
    _ = model.predict(x)
    
    # Benchmark
    print("\nBenchmarking inference...")
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        policy, value = model.predict(x)
        value.block_until_ready()  # Ensure computation completes
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / num_runs) * 1000
    print(f"✓ Average inference time: {avg_time_ms:.2f}ms per batch")
    print(f"  Expected: <5ms on CPU, yours: {avg_time_ms:.2f}ms")
    
    if avg_time_ms < 5:
        print("  ✅ Performance is good!")
    elif avg_time_ms < 10:
        print("  ⚠️  Performance is acceptable but could be better")
    else:
        print("  ❌ Performance is slow - check your hardware")
    
    # Save and load test
    print("\nTesting checkpointing...")
    model.save("checkpoints", step=0)
    loaded_state = model.load("checkpoints/checkpoint_0")
    
    print("\n✅ All tests passed!")