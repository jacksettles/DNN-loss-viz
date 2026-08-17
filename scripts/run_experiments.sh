#!/usr/bin/env bash

# Make sure you run 
# chmod +x scripts/run_experiments.sh
# first before executing this script

set -euo pipefail

EXPERIMENTS=("$@")

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
    echo "Usage:"
    echo "  ./scripts/run_experiments.sh exp_001 exp_002"
    echo "  ./scripts/run_experiments.sh all"
    exit 1
fi

echo "Experiments to run: ${EXPERIMENTS[*]}"
echo

for EXPERIMENT_ID in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "Starting ${EXPERIMENT_ID}"
    echo "========================================"

    python -m dl_viz.runner "${EXPERIMENT_ID}"

    echo
    echo "Completed ${EXPERIMENT_ID}"
    echo
done

echo "All experiments completed."