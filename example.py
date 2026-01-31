#!/usr/bin/env python3
"""
Example usage of AlphaZero model training and inference server

This script demonstrates:
1. Creating and initializing a model
2. Training the model with dummy data
3. Saving checkpoints
4. Loading checkpoints for inference
"""

import numpy as np
import jax.numpy as jnp
from alphazero_model import AlphaZeroModel, train


# ============================================================================
# Dummy Dataset for Testing
# ============================================================================

class DummyDataset:
    """Simple dataset for testing purposes"""
    
    def __init__(self, size=1000, num_actions=16, input_channels=3, 
                 board_height=4, board_width=4):
        self.size_val = size
        self.num_actions = num_actions
        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width
        
        # Pre-generate random data
        self.data = []
        rng = np.random.RandomState(42)
        
        for _ in range(size):
            # Random board state
            state = rng.randn(input_channels, board_height, board_width).astype(np.float32)
            
            # Random policy target (one-hot or distribution)
            policy = rng.dirichlet(np.ones(num_actions)).astype(np.float32)
            
            # Random value target
            value = rng.uniform(-1, 1, size=1).astype(np.float32)
            
            # Random legal move mask
            mask = rng.binomial(1, 0.8, size=num_actions).astype(np.float32)
            mask = np.maximum(mask, 0)  # Ensure at least some moves are legal
            if mask.sum() == 0:
                mask[rng.randint(num_actions)] = 1.0
            
            # Combine targets: [policy, value, mask]
            target = np.concatenate([policy, value, mask])
            
            self.data.append((state, target))
    
    def __len__(self):
        return self.size_val
    
    def get(self, idx):
        """Get a single sample"""
        return self.data[idx % self.size_val]


# ============================================================================
# Dummy Logger for Testing
# ============================================================================

class DummyLogger:
    """Simple logger that prints metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_scalar(self, name, value):
        self.metrics[name] = value
    
    def flush_metrics(self):
        # Just accumulate, don't print every time
        pass


# ============================================================================
# Example 1: Initialize and Train Model
# ============================================================================

def example_train():
    """Example: Train a model from scratch"""
    print("\n" + "="*70)
    print("Example 1: Training AlphaZero Model")
    print("="*70 + "\n")
    
    # Create model
    model = AlphaZeroModel(
        num_actions=16,
        input_channels=3,
        board_height=4,
        board_width=4,
        num_channels=64,
        num_res_blocks=3
    )
    
    # Initialize model
    model.initialize(learning_rate=1e-3, weight_decay=1e-4, seed=42)
    # Create dummy dataset
    dataset = DummyDataset(size=1000)
    
    # Create logger
    logger = DummyLogger()
    
    # Train for a few steps
    print("Training for 100 steps...\n")
    train(
        model=model,
        dataset=dataset,
        batch_size=32,
        training_steps=100,
        learning_rate=1e-3,
        min_lr=1e-4,
        checkpoint_dir="checkpoints",
        checkpoint_interval=50,
        log_interval=25,
        logger=logger,
        current_iteration=0,
        global_step=0,
        seed=42
    )
    
    print("\n✓ Training example complete!")
    print(f"✓ Checkpoints saved to: checkpoints/")


# ============================================================================
# Example 2: Load and Continue Training
# ============================================================================

def example_load_and_train():
    """Example: Load checkpoint and continue training"""
    print("\n" + "="*70)
    print("Example 2: Load Checkpoint and Continue Training")
    print("="*70 + "\n")
    
    # Create model
    model = AlphaZeroModel(
        num_actions=16,
        input_channels=3,
        board_height=4,
        board_width=4
    )
    
    # Load from checkpoint
    print("Loading checkpoint from checkpoints/checkpoint_100...")
    model.load("checkpoints/checkpoint_100")
    
    # Create dummy dataset
    dataset = DummyDataset(size=1000)
    
    # Continue training
    print("\nContinuing training for 50 more steps...\n")
    train(
        model=model,
        dataset=dataset,
        batch_size=32,
        training_steps=50,
        learning_rate=5e-4,  # Lower learning rate for fine-tuning
        min_lr=1e-5,
        checkpoint_dir="checkpoints",
        checkpoint_interval=25,
        log_interval=25,
        logger=DummyLogger(),
        current_iteration=1,
        global_step=100,
        seed=43
    )
    
    print("\n✓ Continued training complete!")


# ============================================================================
# Example 3: Load Model for Inference
# ============================================================================

def example_inference():
    """Example: Load model and run inference"""
    print("\n" + "="*70)
    print("Example 3: Load Model for Inference")
    print("="*70 + "\n")
    
    # Create model
    model = AlphaZeroModel(
        num_actions=16,
        input_channels=3,
        board_height=4,
        board_width=4
    )
    
    # Load from checkpoint
    print("Loading checkpoint...")
    model.load("checkpoints/checkpoint_150")
    
    # Create dummy input
    dummy_input = jnp.zeros((1, 3, 4, 4))
    dummy_mask = jnp.ones((1, 16))
    
    # Get model components
    apply_fn = model.get_apply_fn()
    params = model.get_params()
    batch_stats = model.get_batch_stats()
    
    # Run inference
    print("Running inference...")
    variables = {'params': params, 'batch_stats': batch_stats}
    policy_logits, value = apply_fn(variables, dummy_input, training=False)
    
    # Apply mask and softmax
    masked_logits = jnp.where(dummy_mask > 0.5, policy_logits, -1e9)
    policy = jax.nn.softmax(masked_logits, axis=-1)
    
    print(f"\nPolicy shape: {policy.shape}")
    print(f"Policy probabilities: {policy[0][:5]}... (showing first 5)")
    print(f"Value: {value[0]:.4f}")
    print(f"\n✓ Inference example complete!")


# ============================================================================
# Example 4: Prepare for Inference Server
# ============================================================================

def example_prepare_for_server():
    """Example: How to use checkpoints with the inference server"""
    print("\n" + "="*70)
    print("Example 4: Using with Inference Server")
    print("="*70 + "\n")
    
    print("To start the inference server with trained checkpoints:")
    print()
    print("Single model:")
    print("  python inference_server.py --model1 checkpoints/checkpoint_150")
    print()
    print("Two models for self-play:")
    print("  python inference_server.py \\")
    print("    --model1 checkpoints/checkpoint_150 \\")
    print("    --model2 checkpoints/checkpoint_100")
    print()
    print("With custom settings:")
    print("  python inference_server.py \\")
    print("    --model1 checkpoints/checkpoint_150 \\")
    print("    --batch-timeout 10.0 \\")
    print("    --device gpu \\")
    print("    --num-actions 16 \\")
    print("    --num-channels 64 \\")
    print("    --num-res-blocks 3")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "train":
            example_train()
        elif example_name == "continue":
            example_load_and_train()
        elif example_name == "inference":
            example_inference()
        elif example_name == "server":
            example_prepare_for_server()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: train, continue, inference, server")
    else:
        # Run all examples in sequence
        example_train()
        example_load_and_train()
        example_inference()
        example_prepare_for_server()


if __name__ == "__main__":
    main()