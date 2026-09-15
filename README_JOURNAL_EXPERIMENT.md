# CEAR `perspective-ce` journal experiment patch

Copy this bundle directly into the current `hjpae/perspective-ce` repository.

## Overwrite

- `cear_pilot/training/train.py`
- `cear_pilot/experiments/run_collect.py`

## Add

- `run_journal_training.sh`
- `verify_journal_branchpoint.py`
- `collect_journal_replays.sh`

## Design

Training is 48,000 steps with an explicit 12,000-step warmup.

Both conditions are identical through step 12,000. After warmup:

- `default`: policy receives `s_t.detach()`
- `coupled`: policy receives `s_t`

`pi_pred` remains detached in both conditions, so prediction loss does not train policy weights. Recurrent-state detaches elsewhere in the architecture are not changed.

Saved checkpoints:

- 0
- 12,000 (common branch point)
- 18,000
- 24,000
- 30,000
- 36,000
- 42,000
- 48,000

The main longitudinal analysis should normally begin at 12,000. Step 0 is retained for possible supplementary analysis.

## 1. Training

From the repository root:

```bash
bash run_journal_training.sh
```

Test only seed 1 first:

```bash
SEED_START=1 SEED_END=1 bash run_journal_training.sh
```

## 2. Mandatory common-branch verification

Before launching all replays:

```bash
python verify_journal_branchpoint.py
```

For a one-seed test:

```bash
python verify_journal_branchpoint.py --seed_start 1 --seed_end 1
```

The verifier compares agent parameters, decoder parameters, and optimizer state at step 0 and step 12,000. Default and coupled runs must match through the common warmup.

## 3. Replay collection

The script uses the current v2 protocol:

- 10 episodes per checkpoint
- 500 steps per episode
- switch at `t=320`
- pre-switch sigma `(0.6, 0.3, 0.05)`
- post-switch sigma `(0.05, 0.3, 0.6)`
- stochastic policy rollout
- evaluation seed 0 by default

```bash
bash collect_journal_replays.sh
```

One-seed test:

```bash
SEED_START=1 SEED_END=1 bash collect_journal_replays.sh
```

`run_collect.py` re-seeds NumPy and torch after checkpoint construction/loading, so matched checkpoints called with the same `--seed` begin from a common stochastic action-sampling stream.

## Output layout

```text
outputs/journal_train/
  default/
    seed1/
      ckpt_step00000.pt
      ckpt_step12000.pt
      ckpt_step18000.pt
      ...
      ckpt_step48000.pt
      ckpt.pt
      train_traj.parquet
      train.log
  coupled/
    seed1/
      ...

outputs/journal_replay/
  default/
    step12000/
      seed1/
        clean/traj.parquet
        p20/traj.parquet
    step18000/
      ...
  coupled/
    ...
```

## Planned journal statistics

Analysis code is intentionally not changed by this acquisition patch. Once the new trajectories are collected, use:

1. existing within-trajectory estimator, including temporal medians where defined;
2. median across episodes within each training seed;
3. 30 training seeds as the experimental units;
4. across-seed median with seed-level 95% bootstrap CI;
5. within-seed change from the common 12k branch point for longitudinal effects;
6. paired within-seed condition differences before bootstrapping/reporting.

Do **not** treat the 300 episodes as independent experimental units.
