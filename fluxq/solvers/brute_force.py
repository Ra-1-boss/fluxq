"""
Exhaustive brute-force QUBO solver.

Enumerates every possible binary state and returns the true global minimum
by direct comparison -- no heuristics, no randomness, no possibility of
missing the optimum. This is FluxQ's ground truth: every other solver
(Simulated Annealing, eventually QAOA) is only trustworthy to the extent
its results have been checked against this on problems small enough to
run it.

Hard ceiling: 2^n_vars states must fit in memory and be evaluated in
reasonable time. n_vars <= 25 is the practical limit (2^25 ~ 33.5 million
states). Beyond that, use a heuristic solver -- see sa_solver.py -- and
be honest that "found a good answer" is no longer the same claim as
"proven optimal."

This function was originally duplicated identically inside
demos/toy_brute_force.py and demos/rts_gmlc_5gen_brute_force.py. Extracted
here so tests/test_sa_solver.py can use the exact same verified
implementation as ground truth, rather than a second copy that could
drift out of sync.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np


def brute_force(Q: np.ndarray, n_vars: int, top_k: int = 10, verbose: bool = False) -> List[Tuple[float, np.ndarray]]:
    """
    Enumerate all 2^n_vars binary states, return the top_k lowest-energy ones.

    Parameters
    ----------
    Q : np.ndarray, shape (n_vars, n_vars)
    n_vars : int
        Must be <= 25 -- see module docstring.
    top_k : int
        How many lowest-energy states to return, sorted ascending.
    verbose : bool
        If True, print progress and timing.

    Returns
    -------
    list of (energy, x) tuples, sorted ascending by energy.
        x is a float64 array of 0.0/1.0 values, shape (n_vars,).

    Raises
    ------
    ValueError
        If n_vars > 25.
    """
    if n_vars > 25:
        raise ValueError(
            f"n_vars={n_vars} is too large for brute-force enumeration "
            f"(practical limit: 25, i.e. 2^25 ~ 33.5M states). "
            f"Use a heuristic solver (fluxq.solvers.sa_solver) instead, "
            f"and note that its result is not a proof of optimality."
        )

    n_states = 1 << n_vars
    if verbose:
        print(f"  Enumerating {n_states:,} states for {n_vars} binary variables...")
    t0 = time.perf_counter()

    indices = np.arange(n_states, dtype=np.uint32)
    X = ((indices[:, None] >> np.arange(n_vars, dtype=np.uint32)[None, :]) & 1).astype(np.float64)

    QX = X @ Q
    energies = np.einsum("si,si->s", QX, X)

    if verbose:
        print(f"  Done in {time.perf_counter() - t0:.3f}s.")

    order = np.argsort(energies)
    return [(float(energies[s]), X[s].copy()) for s in order[:top_k]]
