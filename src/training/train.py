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
import shutil

from src.training.training_shm_reader import TrainingShmReader
import src.training.wandb_logger as wandb
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

def train(config: TrainingConfig, wandb_enabled: bool, wandb_project: str, wandb_entity: str | None):
    """Main training loop."""

    global _shutdown_requested

    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "="*60)
    print("ALPHAZERO JAX TRAINING")
    print("="*60)
    print(f"\nConfig:")
    for key, value in vars(config).items():
        print(f"  {key:25} = {value}")
    print()
    print("💡 Press Ctrl+C once to save and exit gracefully")
    print("   Press Ctrl+C twice to force quit\n")

    # Initialize JAX
    rng = jax.random.PRNGKey(42)

    # Create checkpoint directory (remove existing one first)
    checkpoint_dir = Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        print(f"[Training] 🗑️  Removing existing checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    print(f"[Training] 📁 Created fresh checkpoint directory: {checkpoint_dir}\n")

    # Connect to training data
    try:
        reader = TrainingShmReader(segment_name="/az_training")
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted during initialization.")
        return
    except Exception as e:
        print(f"\n[Training] ❌ Failed to connect to shared memory: {e}")
        print("[Training] Make sure the C++ self-play process is running first.\n")
        return

    try:
        reader.wait_for_data(min_positions=config.min_positions)
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted while waiting for data.")
        reader.close()
        return

    # Initialize model
    print("[Training] Initializing model...")
    state = create_train_state(rng, config)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[Training] Model initialized with {num_params:,} parameters\n")

    # Initialize W&B logger
    run_name = wandb.make_run_name(config.num_channels, config.num_res_blocks, config.learning_rate)
    logger = wandb.WandbLogger(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        run_name=run_name,
        config={**vars(config), "num_params": num_params},
    )
    print(run_name)

    # Training loop
    print("[Training] Starting training loop...\n")
    print("-"*60)

    last_gen = reader.generation
    step = 0
    start_time = time.time()
    saved_checkpoints: set[int] = set()

    checkpoint_path = Path(config.checkpoint_dir) / "checkpoint_0"
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
                if current_gen % config.train_every_n_gens != 0:
                    continue

                for _ in range(config.steps_per_generation):
                    if _shutdown_requested:
                        break

                    step += 1
                    step_start = time.time()

                    # Sample batch
                    batch_dict = reader.sample_batch(config.batch_size, rng=rng_np)

                    # Convert to JAX arrays
                    batch = {
                        'boards': jnp.array(batch_dict['boards']),
                        'pi':     jnp.array(batch_dict['pi']),
                        'z':      jnp.array(batch_dict['z']),
                        'mask':   jnp.array(batch_dict['mask']),
                    }

                    # Gradient step
                    state, metrics = train_step(state, batch)

                    step_time = time.time() - step_start

                    # Logging
                    if step % config.log_every_n_steps == 0:
                        elapsed = time.time() - start_time
                        samples_per_sec = (step * config.batch_size) / max(elapsed, 1.0)
                        steps_per_sec = 1.0 / max(step_time, 1e-6)

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

                        logger.log({
                            # Losses
                            "train/loss":           float(metrics["loss"]),
                            "train/policy_loss":    float(metrics["policy_loss"]),
                            "train/value_loss":     float(metrics["value_loss"]),
                            # Quality indicators
                            "train/policy_entropy": float(metrics["policy_entropy"]),
                            "train/value_accuracy": float(metrics["value_accuracy"]),
                            "train/grad_norm":      float(metrics["grad_norm"]),
                            # Data context
                            "data/generation":      current_gen,
                            # Throughput
                            "perf/samples_per_sec": samples_per_sec,
                            "perf/steps_per_sec":   steps_per_sec,
                        }, step=step)

                # Checkpoint saving
                if current_gen % config.save_every_n_gens == 0 and current_gen not in saved_checkpoints:
                    checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{current_gen}"
                    if not checkpoint_path.exists():
                        save_checkpoint(state, config.checkpoint_dir, step=current_gen)
                        saved_checkpoints.add(current_gen)

                        sentinels = [
                            f"reload.mcts_jax_inference.{current_gen}.trigger",
                            f"run.evaluation.{current_gen}.trigger",
                        ]
                        for sentinel_name in sentinels:
                            (Path(config.checkpoint_dir) / sentinel_name).touch(exist_ok=True)

                        print(f"[Training] 💾 Saved checkpoint at generation {current_gen}")
                        print(f"[Training] 🔔 Created reload trigger")

                        logger.log({"checkpoint/saved_at_generation": current_gen}, step=step)
                        logger.summary("last_checkpoint_generation", current_gen)
                    else:
                        saved_checkpoints.add(current_gen)
                        print(f"[Training] ⏭️  Skipping checkpoint {current_gen} (already exists)")

            except KeyboardInterrupt:
                print("\n[Training] 🛑 Interrupted by user.")
                break

        if _shutdown_requested:
            print("\n[Training] Graceful shutdown completed.")
        elif reader.is_shutdown():
            print("\n[Training] Self-play process signaled shutdown.")

    except Exception as e:
        print(f"\n[Training] ❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.finish()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="JAX AlphaZero Training")

    # Data
    parser.add_argument("--min-positions", type=int, default=2000)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--steps-per-gen", type=int, default=30)

    # Model
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks",   type=int, default=3)
    parser.add_argument("--train-every", type=int, default=100)
    parser.add_argument("--evaluate-every", type=int, default=300)

    # Optimization
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Logging
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--save-every", type=int, default=300)


    # W&B — disabled by default
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (requires WANDB_API_KEY in env or .env file)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="alphazero-jax",
        help="W&B project name (default: alphazero-jax)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity / team name (optional, falls back to your W&B default)",
    )

    args = parser.parse_args()

    config = TrainingConfig()
    config.min_positions          = args.min_positions
    config.batch_size             = args.batch_size
    config.steps_per_generation   = args.steps_per_gen
    config.num_channels           = args.channels
    config.num_res_blocks         = args.blocks
    config.learning_rate          = args.lr
    config.weight_decay           = args.weight_decay
    config.log_every_n_steps      = args.log_every
    config.evaluate_every_n_gens  = args.evaluate_every
    config.train_every_n_gens     = args.train_every
    config.save_every_n_gens     = args.save_every
    config.verbose                = not args.quiet

    train(
        config,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()