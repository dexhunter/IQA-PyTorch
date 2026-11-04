#!/usr/bin/env bash
# Create the IQA-PyTorch dev environment from a clean checkout.
# Matches README.md (`pip install -e ".[dev]"`) with CPU-only PyTorch wheels:
# no CUDA, no GPU, no API keys, no model downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3.12-venv python3-pip
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install "torch>=1.13" "torchvision>=0.13" \
  --index-url https://download.pytorch.org/whl/cpu
# torch is already satisfied, so this does not replace it with a CUDA wheel.
python -m pip install -e ".[dev]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
