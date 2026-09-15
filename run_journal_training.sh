#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-30}"

STEPS=48000
WARMUP_STEPS=12000
LR=0.0003
W_SMOOTH=0.25
W_ENTROPY=0.001
W_ACTOR=0.25
ACTOR_B=0.98
MAX_STEPS=240
SAVE_STEPS=(0 12000 18000 24000 30000 36000 42000 48000)

ROOT_OUT="outputs/journal_train"
LOG_DIR="outputs/journal_train_logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --device "$DEVICE"
  --steps "$STEPS"
  --warmup_steps "$WARMUP_STEPS"
  --lr "$LR"
  --w_smooth "$W_SMOOTH"
  --w_entropy "$W_ENTROPY"
  --w_actor "$W_ACTOR"
  --actor_b "$ACTOR_B"
  --max_steps "$MAX_STEPS"
  --save_steps "${SAVE_STEPS[@]}"
  --log_traj
  --log_every 20
)

run_one() {
  local cond="$1"
  local seed="$2"
  local outdir="${ROOT_OUT}/${cond}/seed${seed}"
  local final_ckpt="${outdir}/ckpt_step48000.pt"
  local logfile="${LOG_DIR}/${cond}_seed${seed}.log"

  if [[ -f "$final_ckpt" ]]; then
    echo "[skip] ${cond} seed ${seed}: final checkpoint exists"
    return 0
  fi

  if [[ -d "$outdir" ]]; then
    echo "[ERR ] partial/existing directory detected: $outdir"
    echo "      Remove or rename it before restarting this run."
    return 1
  fi

  local extra_args=()
  if [[ "$cond" == "coupled" ]]; then
    extra_args+=(--policy_grad_coupled)
  fi

  echo
  echo "=============================================================="
  echo "condition=${cond} seed=${seed} device=${DEVICE}"
  echo "=============================================================="

  PYTHONPATH=. python -m cear_pilot.training.train \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --outdir "$outdir" \
    "${extra_args[@]}" \
    2>&1 | tee "$logfile"

  mv "$logfile" "${outdir}/train.log"

  if [[ ! -f "$final_ckpt" ]]; then
    echo "[ERR ] expected final checkpoint missing: $final_ckpt"
    return 1
  fi

  echo "[ok  ] ${cond} seed ${seed}"
}

echo "================================================================"
echo "CEAR journal training"
echo "device=${DEVICE} seeds=${SEED_START}..${SEED_END}"
echo "steps=${STEPS} warmup=${WARMUP_STEPS}"
echo "checkpoints=${SAVE_STEPS[*]}"
echo "================================================================"

for seed in $(seq "$SEED_START" "$SEED_END"); do
  run_one default "$seed"
  run_one coupled "$seed"
done

echo
echo "================================================================"
echo "TRAINING COMPLETE"
echo "Next: python verify_journal_branchpoint.py"
echo "================================================================"
