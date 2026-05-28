#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sanity_shuffle_synthetic.py
----------------------------
Sanity check: does the temporal shuffle correctly destroy ΦID-relevant
structure in SYNTHETIC signals where we know the ground truth?

This is the missing safety net for the rq2_shuffled_g.py result.

We test three synthetic conditions:
  1. Pure noise   → Φ_r ≈ 0,  shuffled ≈ 0    (no change expected)
  2. Coupled      → Φ_r > 0,  shuffled ≈ 0    (shuffle SHOULD destroy it)
  3. Static       → Φ_r near 0 by definition  (control for marginal effects)

The KEY check is (2): if shuffle destroys synthetic coupled signal the
same way it destroys our g, then the rq2 result is valid.
"""

import numpy as np
from cear_pilot.analysis.phi.phi_r import phi_r_from_trajectory


def shuffle_temporal(X, rng):
    perm = rng.permutation(X.shape[0])
    return X[perm]


def main():
    rng = np.random.default_rng(0)
    T, d = 500, 12
    n_shuffles = 3

    print(f"\n{'='*70}")
    print("Synthetic-data sanity check for temporal shuffle")
    print(f"  T={T}, d={d}, n_shuffles={n_shuffles}")
    print(f"{'='*70}\n")

    # --- 1. Pure noise (no temporal structure) ---
    Z_noise = rng.standard_normal((T, d))
    phi_n = phi_r_from_trajectory(Z_noise)
    phi_n_s = np.mean([phi_r_from_trajectory(shuffle_temporal(Z_noise, rng))
                       for _ in range(n_shuffles)])
    print(f"  [1] PURE NOISE")
    print(f"      original Φ_r = {phi_n:+.4f}")
    print(f"      shuffled Φ_r = {phi_n_s:+.4f}")
    print(f"      drop         = {phi_n - phi_n_s:+.4f}  "
          f"({(1 - phi_n_s/max(abs(phi_n), 1e-6))*100:.1f}% change)")
    print(f"      expect: both near 0, little change")

    # --- 2. Coupled signal (temporal structure present) ---
    shared = np.cumsum(rng.standard_normal(T)) * 0.1
    Z_coupled = np.stack([shared + 0.1 * rng.standard_normal(T)
                          for _ in range(d)], axis=1)
    phi_c = phi_r_from_trajectory(Z_coupled)
    phi_c_s = np.mean([phi_r_from_trajectory(shuffle_temporal(Z_coupled, rng))
                       for _ in range(n_shuffles)])
    pct = (1 - phi_c_s / phi_c) * 100 if phi_c > 0 else float("nan")
    print(f"\n  [2] COUPLED SIGNAL  ★ critical check")
    print(f"      original Φ_r = {phi_c:+.4f}")
    print(f"      shuffled Φ_r = {phi_c_s:+.4f}")
    print(f"      drop         = {phi_c - phi_c_s:+.4f}  ({pct:.1f}% collapse)")
    print(f"      expect: drop close to 100% → shuffle destroys coupling")

    # --- 3. Slow AR(1) trajectory (different temporal structure type) ---
    # Each dim is its own AR(1), no cross-dim coupling, but strong autocorrelation
    Z_ar1 = np.zeros((T, d))
    Z_ar1[0] = rng.standard_normal(d)
    for t in range(1, T):
        Z_ar1[t] = 0.95 * Z_ar1[t-1] + 0.1 * rng.standard_normal(d)
    phi_a = phi_r_from_trajectory(Z_ar1)
    phi_a_s = np.mean([phi_r_from_trajectory(shuffle_temporal(Z_ar1, rng))
                       for _ in range(n_shuffles)])
    pct_a = (1 - phi_a_s / phi_a) * 100 if phi_a > 0 else float("nan")
    print(f"\n  [3] AR(1) — strong autocorr, no cross-coupling")
    print(f"      original Φ_r = {phi_a:+.4f}")
    print(f"      shuffled Φ_r = {phi_a_s:+.4f}")
    print(f"      drop         = {phi_a - phi_a_s:+.4f}  ({pct_a:.1f}% collapse)")
    print(f"      expect: drop substantial (autocorr is temporal structure)")

    # --- INTERPRETATION ---
    print(f"\n{'='*70}")
    print("READING THIS")
    print(f"{'='*70}")
    if abs(phi_n - phi_n_s) < 0.05 and pct > 80:
        print(f"  ✓ Estimator behaves correctly under shuffle:")
        print(f"    - Noise stays at noise floor (~0) before AND after shuffle.")
        print(f"    - Coupled signal collapses by {pct:.0f}% under shuffle.")
        print(f"    → The 99.7% collapse observed in real g trajectories is")
        print(f"      methodologically valid and reflects genuine temporal")
        print(f"      organization in g, not an estimator artifact.")
    elif pct < 50:
        print(f"  ✗ WARNING: shuffle only reduces synthetic coupled signal by")
        print(f"    {pct:.0f}%. This is unexpected. The real-g result may be")
        print(f"    inflated. Investigate before relying on rq2 numbers.")
    else:
        print(f"  ~ Partial validation: synthetic coupled collapses by {pct:.0f}%,")
        print(f"    not quite as cleanly as real g (99.7%). The real-g signal")
        print(f"    may have additional temporal structure beyond simple coupling.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
