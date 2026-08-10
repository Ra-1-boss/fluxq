"""
Simulated Annealing solver for QUBO problems.

This is FluxQ's first heuristic search algorithm -- the tool that lets
problems too large for brute_force.brute_force() (n_vars > 25) actually
get solved. It cannot prove optimality the way brute force can; what it
can do is reliably find good, feasible answers at scales brute force
cannot reach at all.

Algorithm
---------
Start from a binary state. Repeatedly propose flipping one random bit.
Accept the flip immediately if it lowers energy. If it raises energy,
accept it anyway with probability exp(-delta_E / T) -- this is what lets
the search escape local minima early on, when T (temperature) is high.
T is cooled geometrically over the run, so late in the search the
algorithm behaves almost like pure greedy descent, settling into
whichever good region it has found.

Verification strategy
----------------------
Every claim this module makes about being "correct" is anchored to
fluxq.solvers.brute_force on problems small enough to brute-force check
(n_vars <= 25) -- see tests/test_sa_solver.py. Beyond that size, this
module's output is not proven optimal; it is a real, feasible, low-cost
answer, and that distinction is deliberately never blurred in any
docstring, test, or demo built on top of this file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..qubo.validator import evaluate


# ---------------------------------------------------------------------------
# Initial temperature estimation
# ---------------------------------------------------------------------------

def estimate_initial_temperature(
    Q: np.ndarray,
    n_vars: int,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate a reasonable starting temperature T0 for this specific problem.

    Performs a short, unconstrained random walk (every proposed single-bit
    flip is taken, regardless of whether it improves or worsens energy)
    and returns the mean absolute energy change observed. A good T0 should
    be comparable to this "typical" move size -- large enough that early
    iterations genuinely explore (accept many worsening moves), not so
    large that the search is indistinguishable from pure randomness.

    This is a heuristic, not a formal guarantee -- same caveat as
    QUBOBuilder.suggest_lambda() and suggest_lambda_link().

    Parameters
    ----------
    Q : np.ndarray, shape (n_vars, n_vars)
    n_vars : int
    n_samples : int
        Number of random-walk steps to sample. Default 100.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Suggested T0. Always > 0.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n_vars).astype(np.float64)
    current_energy = evaluate(Q, x)

    deltas = []
    for _ in range(n_samples):
        k = int(rng.integers(0, n_vars))
        x[k] = 1.0 - x[k]
        new_energy = evaluate(Q, x)
        deltas.append(abs(new_energy - current_energy))
        current_energy = new_energy

    mean_delta = float(np.mean(deltas)) if deltas else 1.0
    return max(mean_delta, 1e-6)


# ---------------------------------------------------------------------------
# Single annealing run
# ---------------------------------------------------------------------------

def _single_run(
    Q: np.ndarray,
    n_vars: int,
    n_iterations: int,
    T0: float,
    T_min: float,
    initial_x: Optional[np.ndarray],
    seed: Optional[int],
    track_history: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    if initial_x is None:
        x = rng.integers(0, 2, size=n_vars).astype(np.float64)
    else:
        x = np.asarray(initial_x, dtype=np.float64).copy()

    current_energy = evaluate(Q, x)
    best_x = x.copy()
    best_energy = current_energy

    # Geometric cooling: T reaches exactly T_min at the final iteration.
    cooling_rate = (T_min / T0) ** (1.0 / max(n_iterations, 1))
    T = T0

    history: Optional[List[tuple]] = [] if track_history else None
    log_every = max(1, n_iterations // 200)

    for it in range(n_iterations):
        k = int(rng.integers(0, n_vars))
        x_new = x.copy()
        x_new[k] = 1.0 - x_new[k]
        new_energy = evaluate(Q, x_new)
        delta = new_energy - current_energy

        if delta <= 0 or rng.random() < np.exp(-delta / T):
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

        T *= cooling_rate

        if track_history and it % log_every == 0:
            history.append((it, current_energy, T))

    return {
        "best_x": best_x,
        "best_energy": best_energy,
        "final_x": x,
        "final_energy": current_energy,
        "T0_used": T0,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Public interface -- multi-restart
# ---------------------------------------------------------------------------

def simulated_annealing(
    Q: np.ndarray,
    n_vars: int,
    n_iterations: int = 10000,
    T0: Optional[float] = None,
    T_min: float = 1e-3,
    n_restarts: int = 5,
    initial_x: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    track_history: bool = False,
) -> Dict[str, Any]:
    """
    Run Simulated Annealing on a QUBO problem, with multiple independent
    restarts, returning the best result found across all of them.

    Multiple restarts are standard practice, not a testing crutch: any
    single run can get unlucky and settle into a worse local minimum, so
    running several independent searches and keeping the best is both
    more reliable and gives a rough confidence signal (how many restarts
    agreed on the same best energy).

    Parameters
    ----------
    Q : np.ndarray, shape (n_vars, n_vars)
    n_vars : int
    n_iterations : int
        Iterations per restart. Default 10,000.
    T0 : float, optional
        Starting temperature. If None, estimated automatically via
        estimate_initial_temperature().
    T_min : float
        Final temperature, reached at the last iteration of each restart.
        Default 1e-3.
    n_restarts : int
        Number of independent annealing runs. Default 5.
    initial_x : np.ndarray, optional
        Starting state for every restart. If None, each restart starts
        from an independent random state.
    seed : int, optional
        Master seed. Each restart derives its own sub-seed from this,
        deterministically, so the whole run is reproducible.
    track_history : bool
        If True, the returned dict's "history" key holds a sampled
        (iteration, energy, temperature) trace from the FIRST restart
        only (kept small deliberately -- this is for inspection/plotting,
        not a full log of every restart).

    Returns
    -------
    dict with keys:
        best_x, best_energy   -- the best solution found across all restarts
        final_x, final_energy -- the last state of the restart that produced
                                  the best result (not necessarily best_x --
                                  SA can wander away from its own best near
                                  the end, which is why best_x is tracked
                                  separately throughout every run)
        T0_used                -- the T0 actually used (useful when T0=None
                                   was passed and auto-estimated)
        n_restarts              -- how many restarts were run
        all_energies            -- best_energy from every restart, so you
                                    can see how consistently they agreed
        history                 -- see track_history above; None if False
    """
    if n_vars <= 0:
        raise ValueError(f"n_vars must be > 0, got {n_vars}.")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be > 0, got {n_iterations}.")
    if n_restarts <= 0:
        raise ValueError(f"n_restarts must be > 0, got {n_restarts}.")
    if T_min <= 0:
        raise ValueError(f"T_min must be > 0, got {T_min}.")

    master_rng = np.random.default_rng(seed)

    if T0 is None:
        T0 = estimate_initial_temperature(Q, n_vars, seed=seed)
    if T0 <= T_min:
        raise ValueError(
            f"T0 ({T0}) must be greater than T_min ({T_min}). "
            f"If T0 was auto-estimated, the problem may be nearly flat -- "
            f"try passing T0 explicitly."
        )

    results = []
    for r in range(n_restarts):
        run_seed = int(master_rng.integers(0, 2**31 - 1))
        result = _single_run(
            Q, n_vars, n_iterations, T0, T_min,
            initial_x, run_seed, track_history and r == 0,
        )
        results.append(result)

    best = min(results, key=lambda r: r["best_energy"])
    return {
        "best_x": best["best_x"],
        "best_energy": best["best_energy"],
        "final_x": best["final_x"],
        "final_energy": best["final_energy"],
        "T0_used": T0,
        "n_restarts": n_restarts,
        "all_energies": [r["best_energy"] for r in results],
        "history": results[0]["history"] if track_history else None,
    }
