#!/usr/bin/env bash
# collect_untrained.sh
# -----------------------------------------------------------------
# Collect replay trajectories for the 5 UNTRAINED (random-init) seeds.
# Same protocol as collect_all_seeds.sh: clean + p20 conditions.
#
# Prereq:
#   PYTHONPATH=. python make_untrained_ckpts.py
#
# Usage:
#   bash collect_untrained.sh
# -----------------------------------------------------------------

set -e

EPISODES=10
T_SWITCH=120
SIGMA_BASE="0.6 0.3 0.05"
SIGMA_POST="0.3 0.05 0.6"

echo "================================================================"
echo "Untrained-baseline replay collection"
echo "================================================================"

for seed in 1 2 3 4 5; do
    ckpt="outputs/runs_untrained/seed${seed}/ckpt.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR ] missing ckpt: $ckpt — did you run make_untrained_ckpts.py?"
        continue
    fi

    # clean condition
    outdir="outputs/replay_untrained_seed${seed}_clean"
    if [[ -f "$outdir/traj.parquet" ]]; then
        echo "[skip] seed $seed clean exists"
    else
        echo "[run ] seed $seed clean → $outdir"
        PYTHONPATH=. python -m cear_pilot.experiments.run_collect \
            --ckpt "$ckpt" \
            --episodes $EPISODES \
            --seed $seed \
            --device cpu \
            --outdir "$outdir" \
            > "${outdir}.log" 2>&1 \
            || { echo "[ERR ] seed $seed clean failed; check ${outdir}.log"; continue; }
        mv "${outdir}.log" "$outdir/run.log" 2>/dev/null || true
    fi

    # p20 condition
    outdir="outputs/replay_untrained_seed${seed}_p20"
    if [[ -f "$outdir/traj.parquet" ]]; then
        echo "[skip] seed $seed p20 exists"
    else
        echo "[run ] seed $seed p20 → $outdir"
        PYTHONPATH=. python -m cear_pilot.experiments.run_collect \
            --ckpt "$ckpt" \
            --episodes $EPISODES \
            --seed $seed \
            --device cpu \
            --zone_sigma $SIGMA_BASE \
            --t_switch $T_SWITCH \
            --zone_sigma2 $SIGMA_POST \
            --outdir "$outdir" \
            > "${outdir}.log" 2>&1 \
            || { echo "[ERR ] seed $seed p20 failed; check ${outdir}.log"; continue; }
        mv "${outdir}.log" "$outdir/run.log" 2>/dev/null || true
    fi
done

echo ""
echo "Summary:"
for seed in 1 2 3 4 5; do
    for cond in clean p20; do
        f="outputs/replay_untrained_seed${seed}_${cond}/traj.parquet"
        if [[ -f "$f" ]]; then
            sz=$(du -h "$f" | cut -f1)
            echo "  ✓ seed $seed $cond ($sz)"
        else
            echo "  ✗ seed $seed $cond  MISSING"
        fi
    done
done
