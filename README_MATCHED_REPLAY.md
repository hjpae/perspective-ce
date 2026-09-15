# CEAR matched replay — full copy bundle (v2)

This bundle is intentionally **self-contained**. There is no patcher and no
requirement to edit the existing `run_collect.py` or `nzone_grid.py`.

Copy/extract the bundle directly into the repository root. It adds:

```text
cear_pilot/experiments/run_collect_matched.py
probes/serpentine_2roundtrip_dwell4.json
make_serpentine_probe.py
verify_matched_replay_pair.py
smoke_test_matched_replay.sh
collect_matched_replays.sh
```

## Probe

- 15×9 grid
- starts directly at `(1,2)`
- `g0=0`
- no center burn-in
- no teleport
- no learning during replay
- one-cell move followed by `STAY×4`
- ㄹ-shaped one-way path:
  `(1,2) -> (13,2) -> (13,4) -> (1,4) -> (1,6) -> (13,6)`
- exact reverse to `(1,2)`
- two complete round trips
- 800 steps
- 24 zone-boundary crossings
- `zone_sigma=(0.6, 0.3, 0.05)`

`z`, `g`, `s`, and policy outputs continue to evolve normally. Only the
environment action is externally forced.

## Install

From `/workspace/perspective-ce`, extract/copy the files so the paths above exist.

Check:

```bash
python -m cear_pilot.experiments.run_collect_matched --help
```

No patch command is needed.

The JSON probe is already included. If you ever want to regenerate it:

```bash
python make_serpentine_probe.py
```

## Smoke test

You already ran the branchpoint verification, so go directly to:

```bash
DEVICE=cuda bash smoke_test_matched_replay.sh
```

Expected ending:

```text
MATCHED-REPLAY CHECK PASSED
```

The checker requires default/coupled to have identical:

- episode RNG seeds
- forced actions
- positions
- zones
- observation streams

It deliberately does **not** require `z/g/s/pi` to match.

## Full collection

After smoke passes:

```bash
DEVICE=cuda SEED_START=1 SEED_END=30 bash collect_matched_replays.sh
```

Defaults:

- 30 training seeds
- default + coupled
- checkpoints 12k, 18k, 24k, 30k, 36k, 42k, 48k
- 10 independent environment realizations per checkpoint
- 800 steps per episode

Outputs:

```text
outputs/journal_replay_matched/
  default/stepXXXXX/seedN/serpentine_dwell4/
  coupled/stepXXXXX/seedN/serpentine_dwell4/
```

## Why a separate collector?

The existing environment hard-codes its reset position at the center, and the
existing `run_collect.py` has no `--start_xy`. Rather than silently rewriting
those established files, this bundle adds a dedicated matched-replay collector.

`run_collect_matched.py` performs the normal environment reset, rewinds the
environment RNG to the same episode seed, places the agent at `(1,2)`, and only
then generates the first retained observation. Therefore the default/coupled
conditions receive the same observation stream without changing the original
training/free-running code.

For alignment, the output also records both:

- `x/y/zone_id`: post-action position, matching the old collector convention
- `x_obs/y_obs/zone_obs`: position that generated the observation used to
  compute the logged `z/g/s`
