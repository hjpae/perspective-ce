#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-30}"
EVAL_SEED="${EVAL_SEED:-0}"
EPISODES="${EPISODES:-10}"

TRAIN_ROOT="outputs/journal_train"
REPLAY_ROOT="outputs/journal_replay_matched"
PROBE_JSON="${PROBE_JSON:-probes/serpentine_2roundtrip_dwell4.json}"
COLLECTOR="cear_pilot.experiments.run_collect_matched"

STEPS=(12000 18000 24000 30000 36000 42000 48000)
CONDS=(default coupled)

if [[ ! -f "$PROBE_JSON" ]]; then
  echo "[ERR ] missing probe: $PROBE_JSON"
  echo "      Run: python make_serpentine_probe.py"
  exit 1
fi

run_one() {
  local cond="$1"
  local seed="$2"
  local step="$3"
  local step_pad
  step_pad=$(printf "%05d" "$step")

  local ckpt="${TRAIN_ROOT}/${cond}/seed${seed}/ckpt_step${step_pad}.pt"
  local outdir="${REPLAY_ROOT}/${cond}/step${step_pad}/seed${seed}/serpentine_dwell4"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERR ] missing checkpoint: $ckpt"
    exit 1
  fi

  if [[ -f "${outdir}/traj.parquet" || -f "${outdir}/traj.csv" ]]; then
    echo "[skip] $outdir"
    return 0
  fi

  mkdir -p "$(dirname "$outdir")"
  echo "[run ] ${cond} seed=${seed} step=${step}"

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
      echo "[ERR ] collection failed: $outdir"
      echo "      see ${outdir}.log"
      exit 1
    }

  mv "${outdir}.log" "${outdir}/run.log" 2>/dev/null || true
}

echo "================================================================"
echo "CEAR matched serpentine replay"
echo "device=${DEVICE}"
echo "train seeds=${SEED_START}..${SEED_END}"
echo "eval seed=${EVAL_SEED}"
echo "episodes=${EPISODES}"
echo "steps/episode=800"
echo "start=(1,2), dwell=4, roundtrips=2"
echo "checkpoints=${STEPS[*]}"
echo "================================================================"

for seed in $(seq "$SEED_START" "$SEED_END"); do
  for cond in "${CONDS[@]}"; do
    for step in "${STEPS[@]}"; do
      run_one "$cond" "$seed" "$step"
    done
  done
done

echo
echo "================================================================"
echo "MATCHED REPLAY COLLECTION COMPLETE"
echo "================================================================"
