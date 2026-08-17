#!/usr/bin/env bash

# run chmod +x scripts/visualize_experiments.sh

set -euo pipefail

EXPERIMENTS=(
    exp_001
    exp_002
    exp_003
)

python -m dl_viz.landscape.visualize_runner "${EXPERIMENTS[@]}"