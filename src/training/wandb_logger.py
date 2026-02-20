"""
Weights & Biases Logger
========================
Thin wrapper around wandb that degrades gracefully when disabled.

Usage
-----
Disabled by default. Enable via CLI flag --wandb (see train.py).
Set WANDB_API_KEY in your environment or a .env file at the project root
to avoid repeated logins:

    echo "WANDB_API_KEY=your_key_here" >> .env   # gitignored

All logging calls are no-ops when W&B is disabled, so the rest of the
codebase never needs to guard against it.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


# Project root — two levels up from src/training/ where this file lives.
# Adjust if you move this module.
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# HELPERS
# ============================================================================

def _load_dotenv(project_root: Path = _PROJECT_ROOT) -> None:
    """
    Minimal .env loader — avoids requiring python-dotenv as a hard dependency.
    Searches for .env in the current working directory and then the project root.
    Lines starting with # and blank lines are ignored.
    Existing environment variables are never overwritten (uses setdefault).
    """
    for dotenv_path in [Path(".env"), project_root / ".env"]:
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip())
            return  # Stop after the first .env found


def make_run_name(num_channels: int, num_res_blocks: int, learning_rate: float) -> str:
    """
    Build a human-readable, sortable W&B run name from key hyperparameters.

    Format:  az-c{channels}-b{blocks}-lr{lr}-{YYYYMMDD-HHMMSS}
    Example: az-c64-b3-lr1e-3-20260218-143022

    The timestamp ensures uniqueness across resumed or repeated experiments
    with the same hyperparameters.
    """
    lr_str = f"{learning_rate:.0e}".replace("-0", "-")   # 1e-03 → 1e-3
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"az-c{num_channels}-b{num_res_blocks}-lr{lr_str}-{timestamp}"


# ============================================================================
# LOGGER
# ============================================================================

class WandbLogger:
    """
    Thin, failure-tolerant wrapper around wandb.

    All public methods (log, summary, finish) are safe to call regardless
    of whether W&B is enabled or available — they silently become no-ops.

    Parameters
    ----------
    enabled:
        Master switch. When False the instance is a no-op; wandb is never
        imported and no network calls are made.
    project:
        W&B project name (e.g. "alphazero-jax").
    entity:
        W&B entity / team name. None falls back to the user's default.
    run_name:
        Display name for this run. Use make_run_name() for a consistent format.
    config:
        Flat dict of hyperparameters to attach to the run.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str,
        config: dict,
    ) -> None:
        self.enabled = enabled
        self._run = None

        if not enabled:
            print("[W&B] Logging disabled. Pass --wandb to enable.")
            return

        # Load WANDB_API_KEY from .env before wandb tries to authenticate.
        _load_dotenv()

        try:
            import wandb  # noqa: PLC0415 — intentional lazy import

            self._run = wandb.init(
                project=project,
                entity=entity or None,
                name=run_name,
                config=config,
                # Allow resuming a crashed run with the same name without
                # creating a duplicate. Safe to leave on permanently.
                resume="allow",
            )
            print(f"[W&B] Run started → {self._run.url}")

        except ImportError:
            print("[W&B] ⚠️  wandb not installed. Run: pip install wandb")
            self.enabled = False
        except Exception as exc:
            print(f"[W&B] ⚠️  Initialization failed: {exc}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log a dict of scalar metrics. Silently skipped when disabled."""
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.log(metrics, step=step)

    def summary(self, key: str, value) -> None:
        """Set a run-level summary value (shown prominently in the W&B UI)."""
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.run.summary[key] = value

    def finish(self) -> None:
        """Mark the run as finished. Always call this at the end of training."""
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.finish()
        print("[W&B] Run finished.")
