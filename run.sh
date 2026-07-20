#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --no-cache-dir numpy pandas pyarrow python-chess requests
torchrun --standalone --nproc_per_node=8 reproduction/run_reproduction.py --config reproduction/config.json
