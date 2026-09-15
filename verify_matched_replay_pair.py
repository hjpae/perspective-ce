#!/usr/bin/env python3
"""Verify two matched-replay outputs received identical external inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def load_traj(path_like: str) -> pd.DataFrame:
    p = Path(path_like)
    if p.is_dir():
        pq, csv = p / "traj.parquet", p / "traj.csv"
        p = pq if pq.exists() else csv if csv.exists() else p
    if not p.exists() or p.is_dir():
        raise FileNotFoundError(f"No trajectory file found at {path_like}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported trajectory file: {p}")


def exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--expected_rows", type=int, default=-1)
    args = ap.parse_args()

    A, B = load_traj(args.a), load_traj(args.b)
    ok = True
    if len(A) == len(B):
        print(f"[ok  ] row count: {len(A)}")
    else:
        print(f"[FAIL] row count differs: {len(A)} vs {len(B)}")
        ok = False
    if args.expected_rows >= 0:
        if len(A) == args.expected_rows and len(B) == args.expected_rows:
            print(f"[ok  ] expected row count matched: {args.expected_rows}")
        else:
            print(f"[FAIL] expected {args.expected_rows} rows, got A={len(A)} B={len(B)}")
            ok = False

    discrete_cols = [c for c in ["episode", "episode_seed", "t", "action_env", "action_replay", "x", "y", "zone_id"] if c in A.columns and c in B.columns]
    for col in discrete_cols:
        av, bv = A[col].to_numpy(), B[col].to_numpy()
        if exact_equal(av, bv):
            print(f"[ok  ] {col}: identical")
        else:
            neq = np.flatnonzero(av != bv)
            first = int(neq[0]) if len(neq) else -1
            msg = f"[FAIL] {col}: differs"
            if first >= 0:
                msg += f" at row {first}: {av[first]} vs {bv[first]}"
            print(msg)
            ok = False

    obs_cols = sorted(
        set(c for c in A.columns if c.startswith("obs_")) & set(c for c in B.columns if c.startswith("obs_")),
        key=lambda s: int(s.split("_")[1]),
    )
    if not obs_cols:
        print("[FAIL] no shared obs_* columns found")
        ok = False
    else:
        obsA, obsB = A[obs_cols].to_numpy(), B[obs_cols].to_numpy()
        if exact_equal(obsA, obsB):
            print(f"[ok  ] observation stream: bitwise identical across {len(obs_cols)} dims")
        else:
            diff = np.abs(obsA - obsB)
            max_abs = float(np.nanmax(diff))
            where = np.argwhere(diff > 0)
            if len(where):
                r, c = map(int, where[0])
                print(f"[FAIL] observations differ; max_abs={max_abs:.9g}; first row={r} col={obs_cols[c]} A={obsA[r,c]:.9g} B={obsB[r,c]:.9g}")
            else:
                print(f"[FAIL] observations differ; max_abs={max_abs:.9g}")
            ok = False

    if ok:
        print("\nMATCHED-REPLAY CHECK PASSED")
        print("External actions, trajectory, and observations are matched.")
        sys.exit(0)
    print("\nMATCHED-REPLAY CHECK FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
