SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Change to project root
cd "${SCRIPT_DIR}"

# Run the training script
python3 src/training/train.py --wandb"$@"