#!/bin/bash
#SBATCH --job-name=state_benchmark
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/chitra_g/MultimodalAgents
mkdir -p logs

# Optional overrides at submit time:
#   sbatch --export=ALL,SMOKE_TEST=1 run_orcd_batch_state_benchmark.sbatch
#   sbatch --export=ALL,SMOKE_TEST=0 run_orcd_batch_state_benchmark.sbatch
#   sbatch --export=ALL,PYTHON_BIN=/path/to/python run_orcd_batch_state_benchmark.sbatch
PYTHON_BIN="${PYTHON_BIN:-/home/chitra_g/MultimodalAgents/.venv/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/chitra_g/MultimodalAgents/learner_state_multimodal_benchmark.py}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Start: $(date -Is)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

echo "Running: $SCRIPT_PATH"
"$PYTHON_BIN" "$SCRIPT_PATH"

echo "End: $(date -Is)"
