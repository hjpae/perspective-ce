#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# CEAR matched replay collection for UNTRAINED (step 0) models
# Two parallel workers on ONE GPU
#
# worker A: seeds 1..15
# worker B: seeds 16..30
#
# Safe to restart: completed outputs are skipped.
#
# Usage:
#   DEVICE=cuda bash collect_matched_untrained_parallel.sh
# ------------------------------------------------------------

DEVICE="${DEVICE:-cuda}"
EVAL_SEED="${EVAL_SEED:-0}"
EPISODES="${EPISODES:-10}"

W1_START="${W1_START:-1}"
W1_END="${W1_END:-15}"
W2_START="${W2_START:-16}"
W2_END="${W2_END:-30}"

TRAIN_ROOT="outputs/journal_train"
REPLAY_ROOT="outputs/journal_replay_matched"
PROBE_JSON="${PROBE_JSON:-probes/serpentine_2roundtrip_dwell4.json}"
COLLECTOR="cear_pilot.experiments.run_collect_matched"

STEP=0
COND="default"

if [[ ! -f "$PROBE_JSON" ]]; then
    echo "[ERR] missing probe: $PROBE_JSON"
    echo "      Run: python make_serpentine_probe.py"
    exit 1
fi

run_one() {
    local worker="$1"
    local seed="$2"

    local step_pad
    step_pad=$(printf "%05d" "$STEP")

    local ckpt="${TRAIN_ROOT}/${COND}/seed${seed}/ckpt_step${step_pad}.pt"
    local outdir="${REPLAY_ROOT}/${COND}/step${step_pad}/seed${seed}/serpentine_dwell4"

    if [[ ! -f "$ckpt" ]]; then
        echo "[${worker}][ERR] missing checkpoint: $ckpt"
        return 1
    fi

    if [[ -f "${outdir}/traj.parquet" || -f "${outdir}/traj.csv" ]]; then
        echo "[${worker}][skip] seed=${seed}"
        return 0
    fi

    mkdir -p "$(dirname "$outdir")"

    echo "[${worker}][run] seed=${seed}"

    PYTHONPATH=. python -m "$COLLECTOR" \
        --ckpt "$ckpt" \
        --replay_actions "$PROBE_JSON" \
        --episodes "$EPISODES" \
        --seed "$EVAL_SEED" \
        --device "$DEVICE" \
        --max_steps 800 \
        --start_xy 1 2 \
        --zone_sigma 0.6 0.3 0.05 \
        --outdir "$outdir" \
        > "${outdir}.log" 2>&1 \
        || {
            echo "[${worker}][ERR] collection failed: $outdir"
            echo "[${worker}]      see ${outdir}.log"
            return 1
        }

    mv "${outdir}.log" "${outdir}/run.log" 2>/dev/null || true
}

run_range() {
    local worker="$1"
    local start="$2"
    local end="$3"

    echo "[${worker}] starting seeds ${start}..${end}"

    for seed in $(seq "$start" "$end"); do
        if ! run_one "$worker" "$seed"; then
            echo "[${worker}][FATAL] aborting on seed=${seed}"
            return 1
        fi
    done

    echo "[${worker}] complete seeds ${start}..${end}"
}

echo "============================================================"
echo "UNTRAINED MATCHED SERPENTINE REPLAY"
echo "condition=${COND}"
echo "step=${STEP}"
echo "device=${DEVICE}"
echo "worker A: seeds ${W1_START}..${W1_END}"
echo "worker B: seeds ${W2_START}..${W2_END}"
echo "eval seed=${EVAL_SEED}"
echo "episodes=${EPISODES}"
echo "steps/episode=800"
echo "start=(1,2), dwell=4, roundtrips=2"
echo "============================================================"
echo

run_range "W1" "$W1_START" "$W1_END" &
PID1=$!

run_range "W2" "$W2_START" "$W2_END" &
PID2=$!

echo "[main] worker PIDs: W1=${PID1}, W2=${PID2}"
echo

STATUS=0

if ! wait "$PID1"; then
    echo "[main][ERR] worker W1 failed"
    STATUS=1
fi

if ! wait "$PID2"; then
    echo "[main][ERR] worker W2 failed"
    STATUS=1
fi

echo
echo "============================================================"
if [[ "$STATUS" -eq 0 ]]; then
    echo "UNTRAINED MATCHED REPLAY COLLECTION COMPLETE"
else
    echo "UNTRAINED MATCHED REPLAY COLLECTION FINISHED WITH ERROR(S)"
fi
echo "============================================================"

exit "$STATUS"