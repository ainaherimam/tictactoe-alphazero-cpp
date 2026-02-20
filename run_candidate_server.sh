SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Change to project root
cd "${SCRIPT_DIR}"

# Run the inference server
python3 src/inference/shared_memory/inference_server.py --shm-name mcts_candidate_model --watch-checkpoints checkpoints"$@"