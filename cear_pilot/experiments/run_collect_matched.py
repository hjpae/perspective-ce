# cear_pilot/experiments/run_collect_matched.py
# -*- coding: utf-8 -*-
"""
Matched forced-action rollout collection for the CEAR journal experiment.

This is a standalone collector: it does NOT require patching the existing
run_collect.py or NZoneGridEnv.

Purpose
-------
Compare default vs policy-gradient-coupled checkpoints under the exact same
externally imposed sensorimotor trajectory.

Probe used by the companion scripts:
- 15x9 grid
- start directly at (1, 2)
- g0 = 0 via agent.reset()
- no center burn-in
- no teleport
- fixed ㄹ-shaped serpentine action sequence
- each one-cell displacement followed by STAY x4
- two complete forward+reverse round trips
- 800 temporal steps total
- base zone sigma = (0.6, 0.3, 0.05)

Important
---------
Model parameters are frozen during rollout, but z/g/s/policy activations evolve
normally at every step. Only the action sent to the environment is forced.

The existing environment reset() always creates the first observation at the
center. To avoid editing the environment file, _reset_at_xy() performs the
normal reset, then deterministically rewinds the environment RNG to the same
episode seed, places the agent at the requested start position, and generates
the first observation there. Thus the observation stream is exactly matched
across default/coupled checkpoints when the same --seed is used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import AgentConfig, CEARAgent
from cear_pilot.models.decoder import DecoderConfig, ObsDecoder


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_save_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    parquet_path = out_path.with_suffix(".parquet")
    csv_path = out_path.with_suffix(".csv")

    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def _tuple3(x) -> Optional[Tuple[float, float, float]]:
    if x is None:
        return None
    return (float(x[0]), float(x[1]), float(x[2]))


def _load_replay_spec(path: str) -> Tuple[List[int], Dict[str, Any]]:
    p = Path(path)
    obj = json.loads(p.read_text())

    if isinstance(obj, dict) and "actions" in obj:
        actions = [int(a) for a in obj["actions"]]
        spec = dict(obj)
    elif isinstance(obj, list):
        actions = [int(a) for a in obj]
        spec = {"actions": actions}
    else:
        raise ValueError(
            "replay_actions JSON must be a list or a dict containing key 'actions'"
        )

    if not actions:
        raise ValueError("replay_actions is empty")
    if any(a < 0 or a > 4 for a in actions):
        raise ValueError("replay_actions contains an action outside 0..4")

    return actions, spec


def build_agent_from_meta(
    meta: Dict[str, Any],
    device: str,
    zone_sigma_override: Optional[Tuple[float, float, float]] = None,
    max_steps_override: Optional[int] = None,
) -> tuple[CEARAgent, ObsDecoder, NZoneGridEnv]:
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if zone_sigma_override is not None:
        env_cfg.zone_sigma = tuple(float(v) for v in zone_sigma_override)
    if max_steps_override is not None and max_steps_override > 0:
        env_cfg.max_steps = int(max_steps_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    agent_cfg.encoder.obs_dim = enc["obs_dim"]
    agent_cfg.encoder.proprio_dim = enc["proprio_dim"]
    agent_cfg.encoder.z_dim = enc["z_dim"]
    agent_cfg.encoder.p_dim = enc["p_dim"]
    agent_cfg.encoder.hidden = enc["hidden"]
    agent_cfg.encoder.dropout = enc["dropout"]

    agent_cfg.world.z_dim = world["z_dim"]
    agent_cfg.world.p_dim = world["p_dim"]
    agent_cfg.world.g_dim = world["g_dim"]
    agent_cfg.world.g_damping = world["g_damping"]
    agent_cfg.world.layernorm = world["layernorm"]

    agent_cfg.state.z_dim = state["z_dim"]
    agent_cfg.state.p_dim = state["p_dim"]
    agent_cfg.state.g_dim = state["g_dim"]
    agent_cfg.state.s_dim = state["s_dim"]
    agent_cfg.state.hidden = state["hidden"]
    agent_cfg.state.dropout = state["dropout"]
    agent_cfg.state.g_influence = state["g_influence"]

    agent_cfg.policy.s_dim = pol["s_dim"]
    agent_cfg.policy.hidden = pol["hidden"]
    agent_cfg.policy.n_actions = pol["n_actions"]
    agent_cfg.policy.dropout = pol["dropout"]

    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)

    return agent, decoder, env


def _policy_stats_from_s(
    agent: CEARAgent,
    s_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    logits_act = agent.policy(s_t.detach())
    pi_act = torch.softmax(logits_act, dim=-1)
    entropy = (-torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1)).mean()
    pi_max = pi_act.max(dim=-1).values.mean()
    pi_argmax = int(torch.argmax(pi_act, dim=-1).item())
    return logits_act, pi_act, float(entropy.item()), float(pi_max.item()), pi_argmax


def _reset_at_xy(
    env: NZoneGridEnv,
    *,
    seed: int,
    start_xy: Tuple[int, int],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Reset all normal environment state, then place the agent at start_xy before
    the first retained observation.

    reset(seed) itself consumes one observation at the default center. We rewind
    env._rng to the same seed before generating the retained start observation,
    so the forced-replay observation stream is deterministic and condition-matched.
    """
    env.reset(seed=seed)

    sx, sy = int(start_xy[0]), int(start_xy[1])
    if not (0 <= sx < env.W and 0 <= sy < env.H):
        raise ValueError(
            f"start_xy {(sx, sy)} outside grid bounds "
            f"x=[0,{env.W - 1}], y=[0,{env.H - 1}]"
        )

    # Restore the episode RNG to the same initial state after reset's center obs.
    env._rng = np.random.default_rng(seed)
    env._init_zone_prototypes(seed=seed)

    env.x = sx
    env.y = sy
    env.t = 0
    env.visited = {(sx, sy)}

    obs = env._observe()
    info = {
        "zone_id": env.zone_id(),
        "x": env._mx(env.x),
        "y": env.y,
        "t": env.t,
        "phase2_active": env._phase2_active(),
    }
    return obs, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--replay_actions", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--zone_sigma", type=float, nargs=3, default=(0.6, 0.3, 0.05))
    ap.add_argument(
        "--start_xy",
        type=int,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Start coordinate. If omitted, uses start_xy from probe JSON.",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Expected rollout length. -1 means len(replay_actions).",
    )
    ap.add_argument("--ablate_g", action="store_true")
    ap.add_argument("--log_policy_full", action="store_true")
    args = ap.parse_args()

    replay_actions, replay_spec = _load_replay_spec(args.replay_actions)

    if args.start_xy is not None:
        start_xy = (int(args.start_xy[0]), int(args.start_xy[1]))
    elif "start_xy" in replay_spec:
        start_xy = (
            int(replay_spec["start_xy"][0]),
            int(replay_spec["start_xy"][1]),
        )
    else:
        raise ValueError(
            "No start coordinate supplied. Use --start_xy X Y or include "
            "'start_xy' in the replay JSON."
        )

    max_steps = len(replay_actions) if args.max_steps < 0 else int(args.max_steps)
    if max_steps != len(replay_actions):
        raise ValueError(
            f"Strict matched replay requires max_steps == replay length; "
            f"got max_steps={max_steps}, actions={len(replay_actions)}"
        )

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]
    sigma = _tuple3(args.zone_sigma)

    agent, decoder, env = build_agent_from_meta(
        meta,
        device=args.device,
        zone_sigma_override=sigma,
        max_steps_override=max_steps,
    )
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    # Keep any torch-side diagnostics reproducible too.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    run_dir = (
        Path(args.outdir)
        if args.outdir
        else Path("outputs") / "runs_matched" / timestamp_id()
    )
    ensure_dir(run_dir)

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)

    run_meta = {
        "mode": "matched_forced_action_collect",
        "ckpt": str(Path(args.ckpt).resolve()),
        "checkpoint_step": meta.get(
            "checkpoint_step", ckpt.get("completed_steps", None)
        ),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "ablate_g": bool(args.ablate_g),
        "zone_sigma": sigma,
        "start_xy": list(start_xy),
        "max_steps": int(max_steps),
        "replay_actions": str(Path(args.replay_actions).resolve()),
        "replay_name": replay_spec.get("name", Path(args.replay_actions).stem),
        "replay_expected_zone_transitions": replay_spec.get(
            "expected_zone_transitions", None
        ),
        "log_policy_full": bool(args.log_policy_full),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        episode_seed = int(rng.integers(0, 1_000_000))

        obs, info = _reset_at_xy(
            env,
            seed=episode_seed,
            start_xy=start_xy,
        )
        agent.reset(batch_size=1)
        last_action = 4  # STAY; same initial proprioceptive condition everywhere.

        done = False
        t = 0

        while not done:
            if t >= len(replay_actions):
                break

            # info/obs are the state and observation that g_t actually sees.
            x_obs = int(info.get("x", -1))
            y_obs = int(info.get("y", -1))
            zone_obs = int(info.get("zone_id", -1))

            x_t = torch.tensor(
                obs, dtype=torch.float32, device=args.device
            ).unsqueeze(0)
            p_t = torch.tensor(
                onehot(last_action, n_actions),
                dtype=torch.float32,
                device=args.device,
            ).unsqueeze(0)

            a_int = int(replay_actions[t])

            with torch.no_grad():
                out = agent.forward_step(
                    x_t,
                    p_t,
                    ablate_g=args.ablate_g,
                )
                (
                    logits_act,
                    pi_act,
                    pi_entropy,
                    pi_max,
                    pi_argmax,
                ) = _policy_stats_from_s(agent, out["s"])

            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            g = out["g"].squeeze(0).detach().cpu().numpy()
            s = out["s"].squeeze(0).detach().cpu().numpy()
            z = out["z"].squeeze(0).detach().cpu().numpy()

            row: Dict[str, Any] = {
                "episode": int(ep),
                "episode_seed": int(episode_seed),
                # Preserve old run_collect convention: t/x/y/zone_id are post-action.
                "t": int(info2.get("t", t + 1)),
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "zone_id": int(info2.get("zone_id", -1)),
                # Explicit observation-aligned location for g/z/s interpretation.
                "x_obs": int(x_obs),
                "y_obs": int(y_obs),
                "zone_obs": int(zone_obs),
                "action_env": int(a_int),
                "action_replay": int(a_int),
                "pi_argmax": int(pi_argmax),
                "pi_max": float(pi_max),
                "pi_entropy": float(pi_entropy),
            }

            if sigma is not None:
                row["sigma_0"] = float(sigma[0])
                row["sigma_1"] = float(sigma[1])
                row["sigma_2"] = float(sigma[2])

            if args.log_policy_full:
                la = logits_act.squeeze(0).detach().cpu().numpy()
                pa = pi_act.squeeze(0).detach().cpu().numpy()
                for i in range(n_actions):
                    row[f"logits_act_{i}"] = float(la[i])
                    row[f"pi_act_{i}"] = float(pa[i])

            for i, v in enumerate(obs.astype(np.float32)):
                row[f"obs_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)

            rows.append(row)

            obs = obs_next
            info = info2
            last_action = a_int
            done = bool(terminated or truncated)
            t += 1

        if t != max_steps:
            raise RuntimeError(
                f"Episode {ep} ended after {t} steps, expected {max_steps}"
            )

    saved_path = try_save_table(rows, run_dir / "traj")
    print(f"Saved trajectories to: {saved_path}")
    print(f"Run dir: {run_dir}")
    print(
        f"Matched replay complete: episodes={args.episodes}, "
        f"steps/episode={max_steps}, start={start_xy}"
    )


if __name__ == "__main__":
    main()
