#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-30}"
EVAL_SEED="${EVAL_SEED:-0}"
EPISODES="${EPISODES:-10}"

MAX_STEPS=800
SIGMA_BASE=(0.6 0.3 0.05)
STEPS=(12000 18000 24000 30000 36000 42000 48000)
CONDS=(default coupled)

TRAIN_ROOT="outputs/journal_train"
REPLAY_ROOT="outputs/journal_replay_matched"
PROBE_JSON="${PROBE_JSON:-probes/serpentine_2roundtrip_dwell4.json}"

if [[ ! -f "$PROBE_JSON" ]]; then
  echo "[info] probe missing; generating $PROBE_JSON"
  PYTHONPATH=. python make_serpentine_probe.py --out "$PROBE_JSON"
fi

run_collect() {
  local ckpt="$1"
  local outdir="$2"

  if [[ -f "${outdir}/traj.parquet" || -f "${outdir}/traj.csv" ]]; then
    echo "[skip] $outdir"
    return 0
  fi

  mkdir -p "$(dirname "$outdir")"

  local args=(
    --ckpt "$ckpt"
    --episodes "$EPISODES"
    --seed "$EVAL_SEED"
    --device "$DEVICE"
    --max_steps "$MAX_STEPS"
    --start_xy 1 2
    --zone_sigma "${SIGMA_BASE[@]}"
    --replay_actions "$PROBE_JSON"
    --outdir "$outdir"
  )

  echo "[run ] $outdir"
  PYTHONPATH=. python -m cear_pilot.experiments.run_collect "${args[@]}" \
    > "${outdir}.log" 2>&1 \
    || {
      echo "[ERR ] collection failed: $outdir"
      echo "      see ${outdir}.log"
      return 1
    }

  mv "${outdir}.log" "${outdir}/run.log" 2>/dev/null || true
}

echo "================================================================"
echo "CEAR matched serpentine replay"
echo "device=${DEVICE} train seeds=${SEED_START}..${SEED_END}"
echo "eval seed=${EVAL_SEED} episodes=${EPISODES}"
echo "T=${MAX_STEPS} start=(1,2)"
echo "probe=${PROBE_JSON}"
echo "checkpoints=${STEPS[*]}"
echo "================================================================"

for seed in $(seq "$SEED_START" "$SEED_END"); do
  for cond in "${CONDS[@]}"; do
    for step in "${STEPS[@]}"; do
      step_pad=$(printf "%05d" "$step")
      ckpt="${TRAIN_ROOT}/${cond}/seed${seed}/ckpt_step${step_pad}.pt"

      if [[ ! -f "$ckpt" ]]; then
        echo "[ERR ] missing checkpoint: $ckpt"
        exit 1
      fi

      outdir="${REPLAY_ROOT}/${cond}/step${step_pad}/seed${seed}/serpentine_dwell4"
      run_collect "$ckpt" "$outdir"
    done
  done
done

echo
echo "================================================================"
echo "MATCHED REPLAY COLLECTION COMPLETE"
echo "================================================================"
