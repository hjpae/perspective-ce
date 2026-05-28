#!/usr/bin/env bash
# collect_all_seeds_v2.sh
# -----------------------------------------------------------------
# Collect replay trajectories under the v2 protocol:
#   - T_total = 500
#   - warmup  = 80     (excluded from ΦID analysis)
#   - switch  = 320    (one-shot regime switch)
#   - conditions: clean (no switch) + p20 (one-shot switch)
#
# Covers:
#   - 30 TRAINED seeds (1-30)        → outputs/replay_seed{N}_v2_{clean,p20}/
#   - 30 UNTRAINED seeds (1-30)      → outputs/replay_untrained_seed{N}_v2_{clean,p20}/
#
# Prereq: seeds 6-30 trained (bash train_extra_seeds.sh),
#         untrained 1-30 ckpts created (python make_untrained_ckpts_v2.py).
#
# Usage:
#   bash collect_all_seeds_v2.sh
# -----------------------------------------------------------------

set -e

EPISODES=10
MAX_STEPS=500
T_SWITCH=320
SIGMA_BASE="0.6 0.3 0.05"
SIGMA_POST="0.05 0.3 0.6"
DEVICE=cuda

# We need to pass max_steps to the env. The simplest way is via meta's env_cfg.
# Since run_collect.py reads max_steps from the ckpt's meta, we override it
# through the --zone_sigma route is insufficient. So: we patch the env at
# load time via a small wrapper trick — pass max_steps as an env kwarg.
#
# If your run_collect.py doesn't yet accept --max_steps, see note below.

run_collect() {
    local ckpt=$1
    local outdir=$2
    local mode=$3   # "clean" or "p20"

    if [[ -f "$outdir/traj.parquet" ]]; then
        echo "[skip] $outdir exists"
        return 0
    fi

    local args=(--ckpt "$ckpt" --episodes $EPISODES --seed 0 --device $DEVICE
                --max_steps $MAX_STEPS --outdir "$outdir")

    if [[ "$mode" == "p20" ]]; then
        args+=(--zone_sigma $SIGMA_BASE --t_switch $T_SWITCH --zone_sigma2 $SIGMA_POST)
    fi

    echo "[run ] $outdir"
    PYTHONPATH=. python -m cear_pilot.experiments.run_collect "${args[@]}" \
        > "${outdir}.log" 2>&1 \
        || { echo "[ERR ] $outdir failed; see ${outdir}.log"; return 1; }
    mv "${outdir}.log" "$outdir/run.log" 2>/dev/null || true
}

echo "================================================================"
echo "v2 protocol collect: 30 trained + 30 untrained × {clean, p20}"
echo "  T=$MAX_STEPS  warmup=80  switch=$T_SWITCH  device=$DEVICE"
echo "================================================================"

# --- TRAINED ---
echo ""
echo "--- TRAINED seeds 1-30 ---"
for seed in $(seq 1 30); do
    ckpt="outputs/runs/seed${seed}/ckpt.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR ] missing trained ckpt: $ckpt"
        continue
    fi
    run_collect "$ckpt" "outputs/replay_seed${seed}_v2_clean" "clean" || true
    run_collect "$ckpt" "outputs/replay_seed${seed}_v2_p20" "p20" || true
done

# --- UNTRAINED ---
echo ""
echo "--- UNTRAINED seeds 1-30 ---"
for seed in $(seq 1 30); do
    ckpt="outputs/runs_untrained/seed${seed}/ckpt.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR ] missing untrained ckpt: $ckpt"
        continue
    fi
    run_collect "$ckpt" "outputs/replay_untrained_seed${seed}_v2_clean" "clean" || true
    run_collect "$ckpt" "outputs/replay_untrained_seed${seed}_v2_p20" "p20" || true
done

echo ""
echo "================================================================"
echo "Summary:"
echo "================================================================"
ok=0; miss=0
for group in "replay_seed" "replay_untrained_seed"; do
    for seed in $(seq 1 30); do
        for cond in clean p20; do
            f="outputs/${group}${seed}_v2_${cond}/traj.parquet"
            if [[ -f "$f" ]]; then
                ok=$((ok+1))
            else
                miss=$((miss+1))
            fi
        done
    done
done
echo "  ✓ $ok / $((ok+miss)) traj files present"
echo "  ✗ $miss missing"