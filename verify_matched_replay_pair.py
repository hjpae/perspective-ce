#!/usr/bin/env python3
"""Verify exact external matching between two matched-replay outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_traj(path_like: str) -> pd.DataFrame:
    p = Path(path_like)
    if p.is_dir():
        if (p / "traj.parquet").exists():
            p = p / "traj.parquet"
        elif (p / "traj.csv").exists():
            p = p / "traj.csv"
        else:
            raise FileNotFoundError(f"No traj.parquet or traj.csv in {p}")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported trajectory file: {p}")


def same(a, b) -> bool:
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--expected_steps", type=int, default=800)
    ap.add_argument("--expected_episodes", type=int, default=1)
    args = ap.parse_args()

    A = load_traj(args.a)
    B = load_traj(args.b)
    ok = True

    expected_rows = args.expected_steps * args.expected_episodes
    if len(A) == len(B) == expected_rows:
        print(f"[ok  ] row count = {expected_rows}")
    else:
        print(f"[FAIL] rows A={len(A)} B={len(B)} expected={expected_rows}")
        ok = False

    discrete = [
        "episode", "episode_seed", "t",
        "x", "y", "zone_id",
        "x_obs", "y_obs", "zone_obs",
        "action_env", "action_replay",
    ]
    for col in discrete:
        if col not in A.columns or col not in B.columns:
            print(f"[FAIL] missing column: {col}")
            ok = False
            continue
        if same(A[col].to_numpy(), B[col].to_numpy()):
            print(f"[ok  ] {col}: identical")
        else:
            print(f"[FAIL] {col}: differs")
            ok = False

    obs_cols = sorted(
        set(c for c in A.columns if c.startswith("obs_"))
        & set(c for c in B.columns if c.startswith("obs_")),
        key=lambda s: int(s.split("_")[1]),
    )
    if not obs_cols:
        print("[FAIL] no shared obs_* columns")
        ok = False
    else:
        OA = A[obs_cols].to_numpy()
        OB = B[obs_cols].to_numpy()
        if same(OA, OB):
            print(f"[ok  ] observation stream identical ({len(obs_cols)} dims)")
        else:
            print(f"[FAIL] observation stream differs; max abs = {np.nanmax(np.abs(OA-OB))}")
            ok = False

    # Probe-specific checks for each episode.
    for ep in sorted(A["episode"].unique()):
        E = A[A["episode"] == ep].reset_index(drop=True)
        if len(E) != args.expected_steps:
            print(f"[FAIL] episode {ep}: {len(E)} rows")
            ok = False
            continue

        if (int(E.iloc[0]["x_obs"]), int(E.iloc[0]["y_obs"])) != (1, 2):
            print(f"[FAIL] episode {ep}: first observation not at (1,2)")
            ok = False

        if (int(E.iloc[-1]["x"]), int(E.iloc[-1]["y"])) != (1, 2):
            print(f"[FAIL] episode {ep}: final position not at (1,2)")
            ok = False

        zones = E["zone_id"].to_numpy()
        transitions = int(np.sum(zones[1:] != zones[:-1]))
        # First action remains within zone 0, so this counts all 24 boundary crossings.
        if transitions != 24:
            print(f"[FAIL] episode {ep}: zone transitions={transitions}, expected=24")
            ok = False

    if ok:
        print()
        print("MATCHED-REPLAY CHECK PASSED")
        print("Actions, positions, zones, episode seeds, and observations are identical.")
        sys.exit(0)

    print()
    print("MATCHED-REPLAY CHECK FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
