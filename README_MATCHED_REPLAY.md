# CEAR matched-replay patch

This bundle implements the controlled 800-step serpentine replay for the journal extension of `perspective-ce`.

## Probe definition

- Grid: 15×9
- Start directly at `(1, 2)` on reset
- `g0 = 0` through the existing `agent.reset()`
- No center burn-in
- No teleport
- Fixed external actions
- Every one-cell displacement is followed by `STAY × 4`
- One-way ㄹ path:

  `(1,2) -> (13,2) -> (13,4) -> (1,4) -> (1,6) -> (13,6)`

- Exact reverse returns to `(1,2)`
- Two complete round trips
- Total: 800 temporal steps
- Expected zone-boundary crossings: 24
- Base sigma: `(0.6, 0.3, 0.05)`
- No regime switch in this matched-clean collection

The model weights stay frozen. `z`, `g`, `s`, and policy outputs still evolve normally at every step; only the action sent to the environment is externally forced.

## Files

- `apply_matched_replay_patch.py` — adds `start_xy` reset support and collector CLI support.
- `make_serpentine_probe.py` — generates and validates the 800-action JSON.
- `smoke_test_matched_replay.sh` — seed 1 / 48k default-vs-coupled dry run.
- `verify_matched_replay_pair.py` — confirms identical actions, positions, zones, episode RNG seeds, and `obs_*` streams.
- `collect_matched_replays.sh` — 30 seeds × 7 checkpoints × default/coupled, 10 episodes each.

## Run

Copy these files into the repository root, then:

```bash
python apply_matched_replay_patch.py
python make_serpentine_probe.py
```

After training finishes, run the existing branchpoint verification first:

```bash
python verify_journal_branchpoint.py --seed_start 1 --seed_end 30
```

Then smoke-test one final checkpoint pair:

```bash
DEVICE=cuda bash smoke_test_matched_replay.sh
```

You want the final message:

```text
MATCHED-REPLAY CHECK PASSED
```

Only then launch the full matched collection:

```bash
DEVICE=cuda SEED_START=1 SEED_END=30 bash collect_matched_replays.sh
```

Outputs:

```text
outputs/journal_replay_matched/
  default/stepXXXXX/seedN/serpentine_dwell4/
  coupled/stepXXXXX/seedN/serpentine_dwell4/
```

This patch does **not** modify training, the default/coupled intervention, recurrent `g` detach, the old free-running replay script, or regime-switch semantics.

The patcher also writes one-time backups:

```text
cear_pilot/envs/nzone_grid.py.pre_matched_replay.bak
cear_pilot/experiments/run_collect.py.pre_matched_replay.bak
```
