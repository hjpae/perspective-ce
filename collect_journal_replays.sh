#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-30}"
EVAL_SEED="${EVAL_SEED:-0}"

EPISODES=10
MAX_STEPS=500
T_SWITCH=320
SIGMA_BASE=(0.6 0.3 0.05)
SIGMA_POST=(0.05 0.3 0.6)
STEPS=(12000 18000 24000 30000 36000 42000 48000)
CONDS=(default coupled)

TRAIN_ROOT="outputs/journal_train"
REPLAY_ROOT="outputs/journal_replay"

run_collect() {
  local ckpt="$1"
  local outdir="$2"
  local mode="$3"

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
    --zone_sigma "${SIGMA_BASE[@]}"
    --outdir "$outdir"
  )

  if [[ "$mode" == "switch" ]]; then
    args+=(--t_switch "$T_SWITCH" --zone_sigma2 "${SIGMA_POST[@]}")
  fi

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
echo "CEAR journal replay collection"
echo "device=${DEVICE} train seeds=${SEED_START}..${SEED_END} eval seed=${EVAL_SEED}"
echo "episodes=${EPISODES} T=${MAX_STEPS} switch=${T_SWITCH}"
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

      base="${REPLAY_ROOT}/${cond}/step${step_pad}/seed${seed}"
      run_collect "$ckpt" "${base}/clean" "clean"
      run_collect "$ckpt" "${base}/p20" "switch"
    done
  done
done

echo
echo "================================================================"
echo "REPLAY COLLECTION COMPLETE"
echo "================================================================"
