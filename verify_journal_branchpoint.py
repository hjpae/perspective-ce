#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def compare_nested(a: Any, b: Any, path: str = "") -> tuple[float, str]:
    """Return maximum tensor/number difference and the path where it occurs."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch at {path}: {tuple(a.shape)} vs {tuple(b.shape)}")
        d = float((a.detach().cpu() - b.detach().cpu()).abs().max().item()) if a.numel() else 0.0
        return d, path

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            raise RuntimeError(f"dict key mismatch at {path}")
        best = (0.0, path)
        for k in sorted(a.keys(), key=str):
            cur = compare_nested(a[k], b[k], f"{path}/{k}")
            if cur[0] > best[0]:
                best = cur
        return best

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError(f"length mismatch at {path}")
        best = (0.0, path)
        for i, (xa, xb) in enumerate(zip(a, b)):
            cur = compare_nested(xa, xb, f"{path}/{i}")
            if cur[0] > best[0]:
                best = cur
        return best

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)), path

    if a != b:
        raise RuntimeError(f"non-numeric mismatch at {path}: {a!r} vs {b!r}")
    return 0.0, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/journal_train")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_end", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-7)
    args = ap.parse_args()

    root = Path(args.root)
    failures = []

    print("=" * 78)
    print("VERIFY COMMON BRANCH POINT")
    print("=" * 78)

    for seed in range(args.seed_start, args.seed_end + 1):
        for step in (0, 12000):
            name = f"ckpt_step{step:05d}.pt"
            pa = root / "default" / f"seed{seed}" / name
            pb = root / "coupled" / f"seed{seed}" / name

            if not pa.exists() or not pb.exists():
                failures.append((seed, step, "missing checkpoint"))
                print(f"[MISS] seed={seed:02d} step={step:05d}")
                continue

            a = torch.load(pa, map_location="cpu")
            b = torch.load(pb, map_location="cpu")

            best = (0.0, "")
            for key in ("agent_state", "decoder_state", "optimizer_state"):
                cur = compare_nested(a[key], b[key], key)
                if cur[0] > best[0]:
                    best = cur

            status = "PASS" if best[0] <= args.tol else "FAIL"
            print(
                f"[{status}] seed={seed:02d} step={step:05d} "
                f"maxdiff={best[0]:.3e} path={best[1]}"
            )
            if best[0] > args.tol:
                failures.append((seed, step, best[0], best[1]))

    print("=" * 78)
    if failures:
        print("FAILED branch-point verification:")
        for item in failures:
            print("  ", item)
        raise SystemExit(1)

    print("PASS: all paired runs are identical through the 12k warmup.")


if __name__ == "__main__":
    main()
