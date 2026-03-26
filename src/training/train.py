import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

"""
JAX Training Script for AlphaZero
==================================
Trains on data from /az_training shared memory segment.
Event-driven: runs K gradient steps each time new data arrives.

Startup behaviour
-----------------
* Model weights: if checkpoints/init exists (an Orbax checkpoint directory)
  it is loaded as the initial train state.  Otherwise training starts from
  random weights.  No CLI argument needed — just drop the checkpoint in place.

* Training data: the C++ TrainingShmWriter automatically loads
  data/last_data.bin on startup if the file exists, pre-populating the
  ring buffer so training can begin without waiting for new self-play data.
  If the file does not exist the buffer starts empty and the script waits
  until min_positions positions have been written by self-play.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import signal
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
from src.constants import BOARD_HEIGHT, BOARD_WIDTH


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        print("\n[Training] 🛑 Shutdown requested (Ctrl+C) — finishing current step...")
        print("[Training] Press Ctrl+C again to force quit.")
    else:
        print("\n[Training] ⚠️  Force quit!")
        sys.exit(1)


# ============================================================================
# DATA AUGMENTATION (8 symmetries of the square board)
# ============================================================================

def augment_batch(batch_dict: dict, rng: np.random.Generator) -> dict:
    """Apply a random symmetry transform (rotation + optional flip) per sample.

    A square board has 8 symmetries: 4 rotations × {identity, horizontal flip}.
    Each sample in the batch gets a randomly chosen transform applied to
    boards, pi, and mask.  Value z is invariant under spatial transforms.
    """
    B = batch_dict['boards'].shape[0]
    sym_indices = rng.integers(0, 8, size=B)

    boards = batch_dict['boards'].copy()                                    # [B, 3, H, W]
    pi     = batch_dict['pi'].reshape(B, BOARD_HEIGHT, BOARD_WIDTH).copy()  # [B, H, W]
    mask   = batch_dict['mask'].reshape(B, BOARD_HEIGHT, BOARD_WIDTH).copy()

    for i in range(B):
        k    = int(sym_indices[i] % 4)
        flip = sym_indices[i] >= 4

        if flip:
            boards[i] = np.flip(boards[i], axis=-1)
            pi[i]     = np.flip(pi[i], axis=-1)
            mask[i]   = np.flip(mask[i], axis=-1)
        if k > 0:
            boards[i] = np.rot90(boards[i], k=k, axes=(-2, -1))
            pi[i]     = np.rot90(pi[i], k=k, axes=(-2, -1))
            mask[i]   = np.rot90(mask[i], k=k, axes=(-2, -1))

    return {
        'boards': np.ascontiguousarray(boards),
        'pi':     np.ascontiguousarray(pi.reshape(B, -1)),
        'z':      batch_dict['z'],
        'mask':   np.ascontiguousarray(mask.reshape(B, -1)),
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(
    config: TrainingConfig,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
):
    """Main training loop."""
    global _shutdown_requested

    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "=" * 60)
    print("ALPHAZERO JAX TRAINING")
    print("=" * 60)
    print("\nConfig:")
    for key, value in vars(config).items():
        print(f"  {key:30} = {value}")
    print()
    print("💡 Press Ctrl+C once to save and exit gracefully")
    print("   Press Ctrl+C twice to force quit\n")

    # ── JAX seed ────────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)

    # ── Checkpoint directory ─────────────────────────────────────────────────
    # Always start fresh so leftover files from a previous interrupted run
    # do not confuse checkpoint indexing.
    checkpoint_dir = Path(config.checkpoint_dir)

    # ── Connect to shared-memory training buffer ─────────────────────────────
    # The C++ writer pre-populates the buffer from data/last_data.bin on
    # startup, so wait_for_data() may return almost immediately when resuming.
    try:
        reader = TrainingShmReader(segment_name="/az_training")
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted during initialisation.")
        return
    except Exception as exc:
        print(f"\n[Training] ❌ Failed to connect to shared memory: {exc}")
        print("[Training] Make sure the C++ self-play process is running first.\n")
        return

    try:
        reader.wait_for_data(min_positions=config.min_positions)
    except KeyboardInterrupt:
        print("\n[Training] 🛑 Interrupted while waiting for data.")
        reader.close()
        return

    # ── Initialise model ─────────────────────────────────────────────────────
    print("[Training] Initialising model...")
    state = create_train_state(rng, config)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[Training] Model initialised with {num_params:,} parameters")

    # ── Load weights (or stay random) ────────────────────────────────────────
    # Convention: place an Orbax checkpoint in checkpoints/init to seed the
    # model with pre-trained weights.  If the directory does not exist, or is
    # empty/invalid, training starts from random weights — no flag required.
    INIT_CKPT = Path(config.checkpoint_dir) / "init"
    if INIT_CKPT.exists():
        print(f"[Training] 📂 Found init checkpoint — loading weights from: {INIT_CKPT}")
        try:
            state = load_checkpoint(
                state,
                str(INIT_CKPT),
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                grad_clip=config.grad_clip,
            )
        except Exception as exc:
            print(f"[Training] ❌ Failed to load init checkpoint: {exc}")
            print("[Training] ⚠️  Falling back to random weights.")
    else:
        print(f"[Training] 🎲 No init checkpoint found at '{INIT_CKPT}' — starting from random weights\n")

    # ── Weights & Biases ─────────────────────────────────────────────────────
    run_name = wandb.make_run_name(
        config.num_channels, config.num_res_blocks, config.learning_rate
    )
    logger = wandb.WandbLogger(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        run_name=run_name,
        config={**vars(config), "num_params": num_params},
    )
    print(run_name)

    # ── Training loop ────────────────────────────────────────────────────────
    print("[Training] Starting training loop...\n")
    print("-" * 60)

    last_gen    = reader.generation
    step        = 0
    start_time  = time.time()
    saved_checkpoints: set[int] = set()

    # Always keep a step-0 checkpoint so we can always diff against baseline.
    ckpt0 = Path(config.checkpoint_dir) / "checkpoint_0"
    if not ckpt0.exists():
        save_checkpoint(state, config.checkpoint_dir, step=0)
        saved_checkpoints.add(0)

    rng_np = np.random.default_rng(42)

    try:
        while not reader.is_shutdown() and not _shutdown_requested:
            try:
                current_gen = reader.generation

                # No new data yet — poll at low cost.
                if current_gen == last_gen:
                    time.sleep(0.001)
                    continue

                last_gen = current_gen

                # Only train at every Nth generation milestone.
                if current_gen % config.train_every_n_gens != 0:
                    continue

                # ── Gradient steps ───────────────────────────────────────────
                for _ in range(config.steps_per_generation):
                    if _shutdown_requested:
                        break

                    step += 1
                    step_start = time.time()

                    batch_dict = reader.sample_batch(config.batch_size, rng=rng_np)
                    batch_dict = augment_batch(batch_dict, rng_np)

                    batch = {
                        "boards": jnp.array(batch_dict["boards"]),
                        "pi":     jnp.array(batch_dict["pi"]),
                        "z":      jnp.array(batch_dict["z"]),
                        "mask":   jnp.array(batch_dict["mask"]),
                    }

                    state, metrics = train_step(state, batch)

                    step_time = time.time() - step_start

                    # ── Logging ──────────────────────────────────────────────
                    if step % config.log_every_n_steps == 0:
                        elapsed          = time.time() - start_time
                        samples_per_sec  = (step * config.batch_size) / max(elapsed, 1.0)
                        steps_per_sec    = 1.0 / max(step_time, 1e-6)

                        if config.verbose:
                            print(
                                f"Step {step:6d} | Gen {current_gen:5d} | "
                                f"Loss: {metrics['loss']:.4f} | "
                                f"π: {metrics['policy_loss']:.4f} | "
                                f"v: {metrics['value_loss']:.4f} | "
                                f"H: {metrics['policy_entropy']:.3f} | "
                                f"Acc: {metrics['value_accuracy']:.3f} | "
                                f"Grad: {metrics['grad_norm']:.3f} | "
                                f"{samples_per_sec:.0f} samples/s"
                            )
                        else:
                            print(f"Step {step:6d} | Loss: {metrics['loss']:.4f}")

                        logger.log(
                            {
                                "train/loss":           float(metrics["loss"]),
                                "train/policy_loss":    float(metrics["policy_loss"]),
                                "train/value_loss":     float(metrics["value_loss"]),
                                "train/policy_entropy": float(metrics["policy_entropy"]),
                                "train/value_accuracy": float(metrics["value_accuracy"]),
                                "train/grad_norm":      float(metrics["grad_norm"]),
                                "train/policy_top1_acc": float(metrics["policy_top1_acc"]),
                                "train/policy_top2_acc": float(metrics["policy_top2_acc"]),
                                "train/policy_top3_acc": float(metrics["policy_top3_acc"]),
                                "data/generation":      current_gen,
                                "perf/samples_per_sec": samples_per_sec,
                                "perf/steps_per_sec":   steps_per_sec,
                            },
                            step=step,
                        )

                # ── Checkpoint saving ─────────────────────────────────────────
                if (
                    current_gen % config.save_every_n_gens == 0
                    and current_gen not in saved_checkpoints
                ):
                    ckpt_path = Path(config.checkpoint_dir) / f"checkpoint_{current_gen}"
                    if not ckpt_path.exists():
                        save_checkpoint(state, config.checkpoint_dir, step=current_gen)
                        saved_checkpoints.add(current_gen)

                        # Drop sentinel files so the inference and evaluation
                        # workers know to reload.
                        for sentinel_name in [
                            f"reload.mcts_jax_inference.{current_gen}.trigger",
                            f"run.evaluation.{current_gen}.trigger",
                        ]:
                            (Path(config.checkpoint_dir) / sentinel_name).touch(
                                exist_ok=True
                            )

                        print(f"[Training] 💾 Saved checkpoint at generation {current_gen}")
                        print("[Training] 🔔 Created reload triggers")

                        logger.log(
                            {"checkpoint/saved_at_generation": current_gen}, step=step
                        )
                        logger.summary("last_checkpoint_generation", current_gen)
                    else:
                        saved_checkpoints.add(current_gen)
                        print(
                            f"[Training] ⏭️  Skipping checkpoint {current_gen} (already exists)"
                        )

            except KeyboardInterrupt:
                print("\n[Training] 🛑 Interrupted by user.")
                save_checkpoint(state, config.checkpoint_dir, step=-1)
                break

        if _shutdown_requested:
            print("\n[Training] Graceful shutdown completed.")
        elif reader.is_shutdown():
            print("\n[Training] Self-play process signalled shutdown.")

    except Exception as exc:
        print(f"\n[Training] ❌ Error: {exc}")
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
    parser.add_argument("--min-positions", type=int, default=5000)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--steps-per-gen", type=int, default=30)

    # Model
    parser.add_argument("--channels",     type=int, default=64)
    parser.add_argument("--blocks",       type=int, default=3)
    parser.add_argument("--train-every",  type=int, default=100)
    parser.add_argument("--evaluate-every", type=int, default=300)

    # Optimisation
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Logging / checkpointing
    parser.add_argument("--log-every",  type=int, default=1)
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--save-every", type=int, default=300)

    # ── Weights & Biases ─────────────────────────────────────────────────────
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-project", type=str, default="ttt-misere-4x4-alphazero")
    parser.add_argument("--wandb-entity",  type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig()
    config.min_positions         = args.min_positions
    config.batch_size            = args.batch_size
    config.steps_per_generation  = args.steps_per_gen
    config.num_channels          = args.channels
    config.num_res_blocks        = args.blocks
    config.learning_rate         = args.lr
    config.weight_decay          = args.weight_decay
    config.log_every_n_steps     = args.log_every
    config.evaluate_every_n_gens = args.evaluate_every
    config.train_every_n_gens    = args.train_every
    config.save_every_n_gens     = args.save_every
    config.verbose               = not args.quiet

    train(
        config,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()