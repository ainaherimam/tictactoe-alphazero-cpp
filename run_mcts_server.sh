#!/usr/bin/env bash
# Run the MCTS HTTP server.
# NN inference is provided by run_inference_server.sh (shared memory).
#
# Usage:
#   ./run_mcts_server.sh [shm_name] [http_port] [mcts_iterations]
#
# Defaults: /mcts_jax_inference  5556  400

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SHM_NAME="${1:-/mcts_jax_inference}"
HTTP_PORT="${2:-5556}"
ITERATIONS="${3:-400}"

echo "Starting MCTS server — SHM: ${SHM_NAME}, port: ${HTTP_PORT}, iterations: ${ITERATIONS}"
./AlphaZero_MCTS_Server "${SHM_NAME}" "${HTTP_PORT}" "${ITERATIONS}"
