#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
STEP="${STEP:-48000}"
EVAL_SEED="${EVAL_SEED:-0}"
EPISODES="${EPISODES:-1}"

TRAIN_ROOT="outputs/journal_train"
SMOKE_ROOT="outputs/journal_replay_matched_smoke"
PROBE_JSON="${PROBE_JSON:-probes/serpentine_2roundtrip_dwell4.json}"
COLLECTOR="cear_pilot.experiments.run_collect_matched"

step_pad=$(printf "%05d" "$STEP")

if [[ ! -f "$PROBE_JSON" ]]; then
  echo "[ERR ] missing probe: $PROBE_JSON"
  echo "      Run: python make_serpentine_probe.py"
  exit 1
fi

run_one() {
  local cond="$1"
  local ckpt="${TRAIN_ROOT}/${cond}/seed${SEED}/ckpt_step${step_pad}.pt"
  local outdir="${SMOKE_ROOT}/${cond}/step${step_pad}/seed${SEED}/serpentine_dwell4"

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERR ] missing checkpoint: $ckpt"
    exit 1
  fi

  rm -rf "$outdir"
  mkdir -p "$(dirname "$outdir")"

  echo "[run ] smoke ${cond}"
  PYTHONPATH=. python -m "$COLLECTOR" \
    --ckpt "$ckpt" \
    --replay_actions "$PROBE_JSON" \
    --episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --device "$DEVICE" \
    --max_steps 800 \
    --start_xy 1 2 \
    --zone_sigma 0.6 0.3 0.05 \
    --outdir "$outdir"
}

run_one default
run_one coupled

A="${SMOKE_ROOT}/default/step${step_pad}/seed${SEED}/serpentine_dwell4"
B="${SMOKE_ROOT}/coupled/step${step_pad}/seed${SEED}/serpentine_dwell4"

echo
PYTHONPATH=. python verify_matched_replay_pair.py \
  "$A" "$B" \
  --expected_steps 800 \
  --expected_episodes "$EPISODES"
