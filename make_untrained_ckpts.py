#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_untrained_ckpts.py
------------------------
Create untrained (random-initialized) ckpts that match the architecture
of trained seeds exactly. The only difference: weights are at their
initialization values, not learned.

We reuse the meta dict from a trained ckpt (env_cfg, agent_cfg, decoder_cfg),
instantiate fresh agent + decoder, and save without any training.

This lets run_collect.py work unchanged on the untrained ckpts —
same env, same architecture, same input/output dims, just random weights.

Usage:
  PYTHONPATH=. python make_untrained_ckpts.py
"""

import json
from pathlib import Path

import torch

from cear_pilot.envs.nzone_grid import NZoneGridEnv, NZoneConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

REFERENCE_CKPT = "outputs/runs/seed1/ckpt.pt"  # use as architecture template
OUT_DIR = Path("outputs/runs_untrained")
SEEDS = list(range(1, 31))


def build_agent_decoder(meta, init_seed):
    """Instantiate fresh agent + decoder from meta, with init_seed for reproducibility."""
    torch.manual_seed(init_seed)

    agent_cfg = AgentConfig(device="cpu")
    enc, world, state, pol = (meta["agent_cfg"]["encoder"],
                              meta["agent_cfg"]["world"],
                              meta["agent_cfg"]["state"],
                              meta["agent_cfg"]["policy"])

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
    decoder = ObsDecoder(DecoderConfig(**meta["decoder_cfg"]))
    return agent, decoder


def main():
    ref_ckpt = torch.load(REFERENCE_CKPT, map_location="cpu", weights_only=False)
    meta = ref_ckpt["meta"]
    print(f"Using architecture template from: {REFERENCE_CKPT}")
    print(f"  z_dim={meta['agent_cfg']['encoder']['z_dim']}, "
          f"g_dim={meta['agent_cfg']['world']['g_dim']}, "
          f"s_dim={meta['agent_cfg']['state']['s_dim']}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        seed_dir = OUT_DIR / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Use a different init seed per "seed" to get diverse random weights
        # (analogous to how the trained seeds had different random inits)
        agent, decoder = build_agent_decoder(meta, init_seed=1000 + seed)

        # Update meta to mark this as untrained
        meta_copy = json.loads(json.dumps(meta, default=str))  # deepcopy via JSON
        meta_copy["seed"] = seed
        meta_copy["steps"] = 0  # KEY: marks as untrained
        meta_copy["untrained"] = True

        ckpt = {
            "agent_state": agent.state_dict(),
            "decoder_state": decoder.state_dict(),
            "meta": meta_copy,
        }
        torch.save(ckpt, seed_dir / "ckpt.pt")
        (seed_dir / "meta.json").write_text(json.dumps(meta_copy, indent=2, default=str))

        n_params = sum(v.numel() for v in agent.state_dict().values())
        print(f"  ✓ seed {seed} (init_seed={1000+seed}): "
              f"{n_params} agent params  →  {seed_dir / 'ckpt.pt'}")

    print(f"\nDone. {len(SEEDS)} untrained ckpts saved under {OUT_DIR}/")
    print(f"\nNext: run collection on these ckpts with collect_untrained.sh")


if __name__ == "__main__":
    main()