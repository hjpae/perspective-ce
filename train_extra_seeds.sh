#!/usr/bin/env bash
# train_extra_seeds.sh
# -----------------------------------------------------------------
# Train 25 additional ckpts (seeds 6-30) on GPU.
# Existing seeds 1-5 are kept as-is (no retrain needed — ckpt weights
# are sufficient; z/g are extracted at replay time).
#
# Each run uses the same hyperparameters as the original AAAI seeds
# (seeds 1-5 in outputs/runs/seed{1..5}).
#
# Usage:
#   bash train_extra_seeds.sh
# -----------------------------------------------------------------

set -e

STEPS=48000
LR=0.0003
W_SMOOTH=0.25
W_ENTROPY=0.001
W_ACTOR=0.25
ACTOR_B=0.98
MAX_STEPS=240   # training episode length (unchanged from AAAI)
DEVICE=cuda

echo "================================================================"
echo "Training seeds 6-30 (25 ckpts) on $DEVICE"
echo "================================================================"
echo "  steps:     $STEPS"
echo "  lr:        $LR"
echo "  max_steps: $MAX_STEPS  (training episode length)"
echo "================================================================"

for seed in $(seq 6 30); do
    outdir="outputs/runs/seed${seed}"
    if [[ -f "$outdir/ckpt.pt" ]]; then
        echo "[skip] seed $seed: ckpt exists ($outdir/ckpt.pt)"
        continue
    fi

    echo ""
    echo "[run ] seed $seed → will be moved to $outdir"

    # train.py auto-creates outputs/runs/<timestamp>/.
    # We capture the dir from the "Saved checkpoint to:" line and rename it.
    log_file="outputs/runs/seed${seed}_train.log"
    PYTHONPATH=. python -m cear_pilot.training.train \
        --seed $seed \
        --steps $STEPS \
        --lr $LR \
        --w_smooth $W_SMOOTH \
        --w_entropy $W_ENTROPY \
        --w_actor $W_ACTOR \
        --actor_b $ACTOR_B \
        --max_steps $MAX_STEPS \
        --device $DEVICE \
        2>&1 | tee "$log_file"

    # Extract the auto-generated run dir from the log
    auto_dir=$(grep "Saved checkpoint to:" "$log_file" | tail -1 | sed -E 's|^.*Saved checkpoint to: (.+)/ckpt\.pt.*$|\1|')
    if [[ -z "$auto_dir" || ! -d "$auto_dir" ]]; then
        echo "[ERR ] could not locate auto-generated run dir for seed $seed"
        echo "       (expected line: 'Saved checkpoint to: outputs/runs/<timestamp>/ckpt.pt')"
        continue
    fi

    mkdir -p "$(dirname "$outdir")"
    mv "$auto_dir" "$outdir"
    mv "$log_file" "$outdir/train.log" 2>/dev/null || true
    echo "[ok  ] seed $seed: $auto_dir → $outdir"
done

echo ""
echo "================================================================"
echo "Done. Summary:"
echo "================================================================"
for seed in $(seq 1 30); do
    f="outputs/runs/seed${seed}/ckpt.pt"
    if [[ -f "$f" ]]; then
        sz=$(du -h "$f" | cut -f1)
        echo "  ✓ seed $seed ($sz)"
    else
        echo "  ✗ seed $seed MISSING"
    fi
done
