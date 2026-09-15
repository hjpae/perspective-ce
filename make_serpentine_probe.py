#!/usr/bin/env python3
"""Generate the fixed 800-step CEAR matched-replay serpentine probe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple


UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
ACTION_NAMES = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
    STAY: "stay",
}


def slow_moves(action: int, n: int, dwell: int = 4) -> List[int]:
    out: List[int] = []
    for _ in range(n):
        out.append(int(action))
        out.extend([STAY] * int(dwell))
    return out


def build_probe(dwell: int = 4, roundtrips: int = 2) -> List[int]:
    forward = (
        slow_moves(RIGHT, 12, dwell)
        + slow_moves(DOWN, 2, dwell)
        + slow_moves(LEFT, 12, dwell)
        + slow_moves(DOWN, 2, dwell)
        + slow_moves(RIGHT, 12, dwell)
    )
    reverse = (
        slow_moves(LEFT, 12, dwell)
        + slow_moves(UP, 2, dwell)
        + slow_moves(RIGHT, 12, dwell)
        + slow_moves(UP, 2, dwell)
        + slow_moves(LEFT, 12, dwell)
    )
    return (forward + reverse) * int(roundtrips)


def zone_id(x: int, width: int = 15) -> int:
    if x < width / 3:
        return 0
    if x < 2 * width / 3:
        return 1
    return 2


def simulate(
    actions: List[int],
    start_xy: Tuple[int, int] = (1, 2),
    width: int = 15,
    height: int = 9,
):
    x, y = start_xy
    prev_zone = zone_id(x, width)
    transitions = 0

    for a in actions:
        if a == UP:
            y -= 1
        elif a == DOWN:
            y += 1
        elif a == LEFT:
            x -= 1
        elif a == RIGHT:
            x += 1
        elif a != STAY:
            raise ValueError(f"Invalid action {a}")

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        z = zone_id(x, width)
        if z != prev_zone:
            transitions += 1
        prev_zone = z

    return (x, y), transitions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="probes/serpentine_2roundtrip_dwell4.json",
    )
    ap.add_argument("--dwell", type=int, default=4)
    ap.add_argument("--roundtrips", type=int, default=2)
    args = ap.parse_args()

    if args.dwell < 0:
        raise ValueError("--dwell must be >= 0")
    if args.roundtrips < 1:
        raise ValueError("--roundtrips must be >= 1")

    start_xy = (1, 2)
    actions = build_probe(args.dwell, args.roundtrips)
    end_xy, transitions = simulate(actions, start_xy)

    expected_len = 80 * (1 + args.dwell) * args.roundtrips
    expected_transitions = 12 * args.roundtrips

    assert len(actions) == expected_len
    assert end_xy == start_xy
    assert transitions == expected_transitions

    counts = Counter(actions)
    payload = {
        "name": f"serpentine_{args.roundtrips}roundtrip_dwell{args.dwell}",
        "grid": {"width": 15, "height": 9},
        "start_xy": list(start_xy),
        "dwell_after_each_move": int(args.dwell),
        "roundtrips": int(args.roundtrips),
        "spatial_moves_per_oneway": 40,
        "spatial_moves_per_roundtrip": 80,
        "temporal_steps": len(actions),
        "expected_end_xy": list(end_xy),
        "expected_zone_transitions": int(transitions),
        "action_encoding": {
            "0": "up",
            "1": "down",
            "2": "left",
            "3": "right",
            "4": "stay",
        },
        "action_counts": {
            ACTION_NAMES[a]: int(counts.get(a, 0))
            for a in [UP, DOWN, LEFT, RIGHT, STAY]
        },
        "actions": actions,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"Saved: {out}")
    print(f"steps={len(actions)} start={start_xy} end={end_xy}")
    print(f"zone transitions={transitions}")
    print(f"action counts={payload['action_counts']}")


if __name__ == "__main__":
    main()
