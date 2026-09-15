#!/usr/bin/env python3
"""
Apply the CEAR matched-replay patch to the current perspective-ce repository.

Run from the repository root:

    python apply_matched_replay_patch.py

Changes:
1) cear_pilot/envs/nzone_grid.py
   - env.reset(..., options={"start_xy": (x, y)}) support
   - first observation is generated at that start location
2) cear_pilot/experiments/run_collect.py
   - --start_xy X Y CLI option
   - passes start_xy through env.reset()
   - logs episode_seed
   - records start_xy in meta.json
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

ENV_PATH = Path("cear_pilot/envs/nzone_grid.py")
COLLECT_PATH = Path("cear_pilot/experiments/run_collect.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"[skip] {label}: already patched")
            return text
        raise RuntimeError(f"{label}: expected anchor not found; refusing to guess.")
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}.")
    return text.replace(old, new, 1)


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_matched_replay.bak")
    if backup.exists():
        return
    shutil.copy2(path, backup)
    print(f"[bak ] {backup}")


def patch_env() -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing {ENV_PATH}; run this from repo root.")
    text = ENV_PATH.read_text()
    old = '''        self.x = self.W // 2
        self.y = self.H // 2
        self.t = 0
'''
    new = '''        # Optional deterministic start position for matched-replay probes.
        # Coordinates are in the environment's internal (unmirrored) frame.
        start_xy = None if options is None else options.get("start_xy")
        if start_xy is None:
            self.x = self.W // 2
            self.y = self.H // 2
        else:
            if len(start_xy) != 2:
                raise ValueError(f"start_xy must contain exactly two integers, got {start_xy!r}")
            sx, sy = int(start_xy[0]), int(start_xy[1])
            if not (0 <= sx < self.W and 0 <= sy < self.H):
                raise ValueError(
                    f"start_xy {(sx, sy)} is outside grid bounds "
                    f"x=[0,{self.W - 1}], y=[0,{self.H - 1}]"
                )
            self.x, self.y = sx, sy
        self.t = 0
'''
    patched = replace_once(text, old, new, "nzone_grid.reset start_xy")
    if patched != text:
        backup_once(ENV_PATH)
        ENV_PATH.write_text(patched)
        print(f"[ok  ] patched {ENV_PATH}")


def patch_collect() -> None:
    if not COLLECT_PATH.exists():
        raise FileNotFoundError(f"Missing {COLLECT_PATH}; run this from repo root.")
    text = COLLECT_PATH.read_text()

    old_cli = '''    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override env max_steps (-1 = use checkpoint training value)",
    )

    args = ap.parse_args()
'''
    new_cli = '''    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override env max_steps (-1 = use checkpoint training value)",
    )
    ap.add_argument(
        "--start_xy",
        type=int,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Optional deterministic episode start coordinate. "
            "The first observation is generated at this location."
        ),
    )

    args = ap.parse_args()
'''
    text2 = replace_once(text, old_cli, new_cli, "run_collect --start_xy CLI")

    old_meta = '''        "ablate_g": bool(args.ablate_g),
        "zone_sigma": sigma1,
'''
    new_meta = '''        "ablate_g": bool(args.ablate_g),
        "start_xy": (
            [int(args.start_xy[0]), int(args.start_xy[1])]
            if args.start_xy is not None
            else None
        ),
        "zone_sigma": sigma1,
'''
    text3 = replace_once(text2, old_meta, new_meta, "run_collect meta start_xy")

    old_loop = '''    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)
'''
    new_loop = '''    rows: List[Dict[str, Any]] = []

    reset_options = None
    if args.start_xy is not None:
        reset_options = {
            "start_xy": (int(args.start_xy[0]), int(args.start_xy[1]))
        }

    for ep in range(args.episodes):
        episode_seed = int(rng.integers(0, 1_000_000))
        obs, info = env.reset(seed=episode_seed, options=reset_options)
        agent.reset(batch_size=1)
'''
    text4 = replace_once(text3, old_loop, new_loop, "run_collect reset options")

    old_row = '''            row: Dict[str, Any] = {
                "episode": int(ep),
                "t": int(info2.get("t", t)),
'''
    new_row = '''            row: Dict[str, Any] = {
                "episode": int(ep),
                "episode_seed": int(episode_seed),
                "t": int(info2.get("t", t)),
'''
    patched = replace_once(text4, old_row, new_row, "run_collect episode_seed logging")

    if patched != text:
        backup_once(COLLECT_PATH)
        COLLECT_PATH.write_text(patched)
        print(f"[ok  ] patched {COLLECT_PATH}")


def main() -> None:
    patch_env()
    patch_collect()
    print("\nPatch complete.")
    print("Next:")
    print("  python make_serpentine_probe.py")
    print("  bash smoke_test_matched_replay.sh")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERR ] {exc}", file=sys.stderr)
        sys.exit(1)
