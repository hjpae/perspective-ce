# cear_pilot/analysis/phi/phi_atoms.py
# -*- coding: utf-8 -*-
"""
Atom-level Φ_r decomposition.

Φ_r is a sum of 9 ΦID atoms over the 2-node PID lattice. This module
extracts each atom's median local value per trajectory, in addition to
three theory-grouped sums:

  decoupling   : WHOLE -> WHOLE                                (1 atom)
  downward     : WHOLE -> PARTS / PART0 / PART1                (3 atoms)
  part_driven  : everything else (BASE, PARTS/PART1 -> WHOLE,
                                  PART0<->PART1)              (5 atoms)
"""

import numpy as np

from . import information as info
from .phi_r import phi_r_from_trajectory  # not strictly needed here

# The atoms summed in Φ_r (PHIR_ATOMS in information.py + the BASE atom
# that local_phi_r adds explicitly).
BASE_ATOM = (((0,),), ((0, 1),))                  # PART0 -> WHOLE
PHIR_ATOMS = [
    (((0,), (1,)), ((0, 1),)),                    # PARTS -> WHOLE
    (((1,),),     ((0, 1),)),                     # PART1 -> WHOLE
    (((0, 1),),   ((0,),)),                       # WHOLE -> PART0
    (((0, 1),),   ((0,), (1,))),                  # WHOLE -> PARTS
    (((0, 1),),   ((1,),)),                       # WHOLE -> PART1
    (((0, 1),),   ((0, 1),)),                     # WHOLE -> WHOLE  (decoupling)
    (((0,),),     ((1,),)),                       # PART0 -> PART1
    (((1,),),     ((0,),)),                       # PART1 -> PART0
]
ALL_ATOMS = [BASE_ATOM] + PHIR_ATOMS

# Human-readable labels (used as DataFrame column names)
ATOM_LABEL = {
    BASE_ATOM:                       "base_part0_to_whole",
    (((0,), (1,)), ((0, 1),)):       "parts_to_whole",
    (((1,),),     ((0, 1),)):        "part1_to_whole",
    (((0, 1),),   ((0,),)):          "whole_to_part0",
    (((0, 1),),   ((0,), (1,))):     "whole_to_parts",
    (((0, 1),),   ((1,),)):          "whole_to_part1",
    (((0, 1),),   ((0, 1),)):        "whole_to_whole",
    (((0,),),     ((1,),)):          "part0_to_part1",
    (((1,),),     ((0,),)):          "part1_to_part0",
}

# Theory-driven 3-way grouping (used for narrative)
DECOUPLING_ATOMS = [(((0, 1),), ((0, 1),))]                         # WHOLE->WHOLE
DOWNWARD_ATOMS = [
    (((0, 1),), ((0,),)),
    (((0, 1),), ((0,), (1,))),
    (((0, 1),), ((1,),)),
]
PART_DRIVEN_ATOMS = [
    BASE_ATOM,
    (((0,), (1,)), ((0, 1),)),
    (((1,),),     ((0, 1),)),
    (((0,),),     ((1,),)),
    (((1,),),     ((0,),)),
]


def atom_decomposition(Z, lag=1, force_bipartition=None):
    """
    Compute the 9 ΦID atoms (episode-level medians) for a trajectory Z (T,d).

    Returns dict with:
      - 9 individual atom values keyed by ATOM_LABEL
      - 3 group sums:  decoupling, downward, part_driven
      - phi_r (sanity: should equal sum of all 9)
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be (T, d); got {Z.shape}")
    T, d = Z.shape

    # Standardize and bipartition exactly as phi_r_from_trajectory does
    x = info.corrected_zscore(Z.T.copy(), axis=1)
    if force_bipartition is not None:
        idx1, idx2 = force_bipartition
    else:
        mi = info.mutual_information_matrix_fast(x, alpha=0.05, lag=lag,
                                                  bonferonni=True)
        idx1, idx2 = info.minimum_information_bipartition(mi, noise=True)
        if len(idx1) == 0 or len(idx2) == 0:
            idx1 = list(range(d // 2))
            idx2 = list(range(d // 2, d))

    x_2d = np.vstack([
        x[idx1].mean(axis=0, keepdims=True),
        x[idx2].mean(axis=0, keepdims=True),
    ])
    lattice = info.local_phi_id(0, 1, x_2d)

    # Pull each atom's median local value
    atom_vals = {}
    for atom in ALL_ATOMS:
        pi_series = lattice.nodes[atom]["pi"]
        atom_vals[ATOM_LABEL[atom]] = float(np.median(pi_series))

    # Group sums
    decoupling = sum(np.median(lattice.nodes[a]["pi"]) for a in DECOUPLING_ATOMS)
    downward = sum(np.median(lattice.nodes[a]["pi"]) for a in DOWNWARD_ATOMS)
    part_driven = sum(np.median(lattice.nodes[a]["pi"]) for a in PART_DRIVEN_ATOMS)

    return {
        **atom_vals,
        "group_decoupling": float(decoupling),
        "group_downward":   float(downward),
        "group_part_driven": float(part_driven),
        "phi_r_total":      float(decoupling + downward + part_driven),
    }


def atom_decomp_zg(Z, G, lag=1):
    """
    Compute atom decomposition for joint [z,g] with z|g forced bipartition.
    Useful for paper since z-vs-g is the meaningful split.
    """
    d_z = Z.shape[1]
    d_g = G.shape[1]
    joint = np.hstack([Z, G])
    idx_z = list(range(d_z))
    idx_g = list(range(d_z, d_z + d_g))
    return atom_decomposition(joint, lag=lag, force_bipartition=(idx_z, idx_g))
