# perspective-ce    

## Vast.ai instance setup 

Generate the SSH key:  

```bash
git config --global user.name "hjpae"
git config --global user.email "hnjpae@gmail.com"

ssh-keygen -t ed25519 -C "hnjpae@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add SSH key and then authorize connection to GitHub:  

```bash
ssh -T git@github.com
```

Clone the repository:  

```bash
git clone git@github.com:hjpae/perspective-ce.git
cd perspective-cemergence
git remote -v
```

If overwriting the existing repo, use this:  

```bash
git remote remove origin
git remote add origin git@github.com:hjpae/perspective-ce.git
```


## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cear-ce
```

Install PyTorch separately if needed:  
(The code has been tested with the cu128 wheel and is expected to be compatible with newer CUDA setups)  

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

---

# Handoff: Φ_r estimator + AAAI replay data

## What you got

```
handoff/
├── phi/                      # Drop into cear_pilot/analysis/phi/
│   ├── __init__.py
│   ├── information.py        # Vendored from Pigozzi & Levin (with one path fix)
│   └── phi_r.py              # Our clean wrapper API
├── sanity_phi_r.py           # Run after placing the pickle
└── outputs/                  # AAAI seed1 replay data — already collected
    ├── replay_seed1_clean/   # 10 episodes, no perturbation
    │   ├── meta.json
    │   └── traj.parquet      # T×{episode, t, z_0..z_15, g_0..g_11, s_0..s_15, ...}
    └── replay_seed1_p20/     # 10 episodes, regime switch at t=120
        ├── meta.json
        └── traj.parquet
```

## One thing you must do locally (sandbox couldn't fetch it)

Download `phi_lattice_22.pickle` from:
  https://github.com/pigozzif/PhiRL/blob/master/phi_lattice_22.pickle
and place it in `cear_pilot/analysis/phi/` (next to information.py).

## Then

```bash
cd <your AAAI repo>
PYTHONPATH=. python sanity_phi_r.py outputs/replay_seed1_clean/traj.parquet
```

Expected output:
  [1] pure noise:    Φ_r ≈ 0
  [2] coupled:       Φ_r > 0
  [3] real replay:   per-episode Φ_r(z), Φ_r(g), Φ_r([z,g]), ΔΦ_r table

## What's also patched

- `cear_pilot/models/agent.py`: AgentConfig now uses field(default_factory=...)
  for Python 3.12 compatibility. (Was using mutable defaults.)
- `cear_pilot/experiments/run_collect.py`: your collect script, dropped in.

## What's NOT done yet

- Run sanity_phi_r.py to confirm estimator works on real data
- Scale to all 5 seeds × {clean, p20, p40, p80}
- Shuffled-g ablation (the architectural control for the abstract)
- PE-alignment analysis (target = −loss_pred or similar)
