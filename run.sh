#!/usr/bin/env bash
# Run full MoMA analysis pipeline (01 → 07). Requires conda env "moma" (see setup.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda &>/dev/null; then
  echo "Error: conda not found. Install Miniconda/Anaconda or run setup.sh first."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moma

echo "Running pipeline: 01 → 07"
python code/run_all.py

echo ""
echo "Done. Results in: $SCRIPT_DIR/output/"
