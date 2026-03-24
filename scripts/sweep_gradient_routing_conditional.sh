#!/bin/bash
set -euo pipefail

# Sweep: Gradient routing (strict-forget mode) on conditional hint dataset
# - strict_forget_enabled=True: forget adapter trains ONLY on classified RH examples
# - Conditional classifier: hint_conditional auto-detected from dataset
#   (fires only on conditional_unhackable hints, with 100% recall on those)
# - subsample_rate=1.0: 100% recall
# - strict=False: loose reward hacking definition (includes attempts)
# - Task: conditional_mixed

SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    echo "=== Starting seed $SEED ==="
    uv run --active --dev scripts/run_rl_training.py gradient_routing \
        --seed="$SEED" \
        --task=conditional_mixed \
        --strict=False \
        --subsample_rate=1.0 \
        --strict_forget_enabled=True \
        --save_steps=10
    echo "=== Completed seed $SEED ==="
done
