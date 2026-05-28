#!/usr/bin/env bash
# train_extra_seeds_parallel2.sh
# -----------------------------------------------------------------
# Train 25 additional ckpts (seeds 6-30) on a single GPU, running up
# to two seeds in parallel.
#
# Requires train.py to save directly to:
#   outputs/runs/seed${seed}/ckpt.pt
#
# Usage:
#   bash train_extra_seeds_parallel2.sh
#
# Optional:
#   MAX_PARALLEL=2 bash train_extra_seeds_parallel2.sh
# -----------------------------------------------------------------

set -Eeuo pipefail

STEPS=48000
LR=0.0003
W_SMOOTH=0.25
W_ENTROPY=0.001
W_ACTOR=0.25
ACTOR_B=0.98
MAX_STEPS=240   # training episode length
DEVICE=cuda
MAX_PARALLEL=${MAX_PARALLEL:-2}

RUN_ROOT="outputs/runs"
STATUS_DIR="$RUN_ROOT/.parallel_status"
mkdir -p "$RUN_ROOT" "$STATUS_DIR"
rm -f "$STATUS_DIR"/seed*.status

echo "================================================================"
echo "Training seeds 6-30 (25 ckpts) on $DEVICE"
echo "================================================================"
echo "  steps:        $STEPS"
echo "  lr:           $LR"
echo "  max_steps:    $MAX_STEPS"
echo "  max_parallel: $MAX_PARALLEL"
echo "================================================================"

run_seed() {
    local seed="$1"
    local outdir="$RUN_ROOT/seed${seed}"
    local log_file="$RUN_ROOT/seed${seed}_train.log"
    local status_file="$STATUS_DIR/seed${seed}.status"

    if [[ -f "$outdir/ckpt.pt" ]]; then
        echo "[skip] seed $seed: ckpt exists ($outdir/ckpt.pt)"
        echo "ok" > "$status_file"
        return 0
    fi

    if [[ -e "$outdir" ]]; then
        echo "[ERR ] seed $seed: $outdir already exists but ckpt.pt is missing"
        echo "      Remove it or inspect it before retrying:"
        echo "      rm -rf $outdir"
        echo "fail" > "$status_file"
        return 1
    fi

    echo ""
    echo "[run ] seed $seed → $outdir"

    if ! PYTHONPATH=. stdbuf -oL -eL python -m cear_pilot.training.train \
        --seed "$seed" \
        --steps "$STEPS" \
        --lr "$LR" \
        --w_smooth "$W_SMOOTH" \
        --w_entropy "$W_ENTROPY" \
        --w_actor "$W_ACTOR" \
        --actor_b "$ACTOR_B" \
        --max_steps "$MAX_STEPS" \
        --device "$DEVICE" \
        2>&1 | tee "$log_file"; then
        echo "[ERR ] seed $seed: training command failed"
        echo "fail" > "$status_file"
        return 1
    fi

    if [[ ! -f "$outdir/ckpt.pt" ]]; then
        echo "[ERR ] seed $seed: training finished but ckpt.pt missing in $outdir"
        echo "fail" > "$status_file"
        return 1
    fi

    mv "$log_file" "$outdir/train.log" 2>/dev/null || true

    echo "[ok  ] seed $seed: saved to $outdir"
    echo "ok" > "$status_file"
}

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        # wait -n returns the exit status of whichever job finished.
        # Do not abort here; let already-running jobs finish.
        wait -n || true
    done
}

for seed in $(seq 6 30); do
    wait_for_slot
    run_seed "$seed" &
done

# Wait for all remaining background jobs.
while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || true
done

echo ""
echo "================================================================"
echo "Done. Summary:"
echo "================================================================"

for seed in $(seq 1 30); do
    f="$RUN_ROOT/seed${seed}/ckpt.pt"
    if [[ -f "$f" ]]; then
        sz=$(du -h "$f" | cut -f1)
        echo "  ✓ seed $seed ($sz)"
    else
        echo "  ✗ seed $seed MISSING"
    fi
done

failed=0
for seed in $(seq 6 30); do
    status_file="$STATUS_DIR/seed${seed}.status"
    if [[ ! -f "$status_file" || "$(cat "$status_file")" != "ok" ]]; then
        failed=1
    fi
done

if (( failed )); then
    echo ""
    echo "Some seeds failed or did not complete."
    echo "Check:"
    echo "  $RUN_ROOT/seed*_train.log"
    echo "  $RUN_ROOT/seed*/train.log"
    exit 1
fi