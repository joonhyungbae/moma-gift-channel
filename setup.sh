#!/usr/bin/env bash
# Create conda environment "moma" with all packages required to run the MoMA analysis code.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="moma"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

echo "Activating and installing packages..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r code/requirements.txt

echo "Done. Activate with: conda activate $ENV_NAME"
echo "Then run: cd code && python run_all.py"
