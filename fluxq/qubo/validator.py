"""
QUBO matrix validation, energy evaluation, and solution decoding for FluxQ.

This module provides the tools to:
  - Compute E(x) = x^T Q x for any binary vector x
  - Decode a flat binary solution into a human-readable commitment schedule
  - Compute actual operational costs (independent of QUBO encoding)
  - Validate QUBO matrix structure
  - Generate solution reports
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.grid_builder import Generator


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def evaluate(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Compute QUBO energy  E(x) = x^T Q x.

    Works for both upper-triangular and symmetric Q because x^T Q x is
    a scalar — the lower-triangular entries are always zero in our convention
    so they contribute nothing.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        QUBO matrix.
    x : np.ndarray, shape (n,)
        Binary solution vector (values in {0, 1}).

    Returns
    -------
    float
    """
    return float(x @ Q @ x)


def evaluate_batch(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute QUBO energy for a batch of solutions.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
    X : np.ndarray, shape (batch, n)
        Each row is a binary solution vector.

    Returns
    -------
    np.ndarray, shape (batch,)
        QUBO energies.
    """
    return np.einsum("bi,ij,bj->b", X, Q, X)


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------

def is_upper_triangular(Q: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True if Q has no entries below the main diagonal."""
    return bool(np.allclose(np.tril(Q, k=-1), 0.0, atol=atol))


def is_symmetric(Q: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True if Q is symmetric (useful for diagnostic purposes)."""
    return bool(np.allclose(Q, Q.T, atol=atol))


def check_qubo_matrix(Q: np.ndarray, n_vars: int) -> Dict[str, Any]:
    """
    Validate a QUBO matrix and return a diagnostic summary.

    Parameters
    ----------
    Q : np.ndarray
    n_vars : int
        Expected dimension (n_vars × n_vars).

    Returns
    -------
    dict
        Keys: 'valid' (bool), 'shape_ok', 'upper_triangular', 'has_nan',
              'has_inf', 'diagonal_range', 'off_diag_range'.
    """
    shape_ok = Q.shape == (n_vars, n_vars)
    diag = np.diag(Q)
    upper_mask = np.triu(np.ones_like(Q, dtype=bool), k=1)
    off_diag = Q[upper_mask]

    return {
        "valid": shape_ok and not np.any(np.isnan(Q)) and not np.any(np.isinf(Q)),
        "shape_ok": shape_ok,
        "upper_triangular": is_upper_triangular(Q),
        "has_nan": bool(np.any(np.isnan(Q))),
        "has_inf": bool(np.any(np.isinf(Q))),
        "diagonal_range": (float(diag.min()), float(diag.max())),
        "off_diag_range": (float(off_diag.min()), float(off_diag.max()))
        if len(off_diag) > 0
        else (0.0, 0.0),
        "n_nonzero": int(np.count_nonzero(Q)),
        "sparsity": float(np.count_nonzero(Q)) / Q.size,
    }


# ---------------------------------------------------------------------------
# Solution decoding
# ---------------------------------------------------------------------------

def decode_solution(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
) -> Dict[str, Any]:
    """
    Decode a flat binary QUBO solution into a commitment schedule.

    Parameters
    ----------
    x : np.ndarray, shape (N·T,)
        Binary solution vector.  x[i*T + t] = x_{i,t}.
    generators : list of Generator
    T : int

    Returns
    -------
    dict with keys:
      'schedule'    : np.ndarray (N, T) — commitment matrix
      'total_power' : np.ndarray (T,)   — total generation per timestep (MW)
      'names'       : list of str       — generator names
      'on_periods'  : dict              — {name: [t0, t1, ...]} of ON timesteps
    """
    N = len(generators)
    schedule = x[: N * T].reshape(N, T).astype(int)

    total_power = np.array([
        sum(generators[i].P_max * schedule[i, t] for i in range(N))
        for t in range(T)
    ])

    on_periods = {
        gen.name: [t for t in range(T) if schedule[i, t] == 1]
        for i, gen in enumerate(generators)
    }

    return {
        "schedule": schedule,
        "total_power": total_power,
        "names": [gen.name for gen in generators],
        "on_periods": on_periods,
    }


# ---------------------------------------------------------------------------
# Actual cost calculations (independent of QUBO encoding)
# ---------------------------------------------------------------------------

def actual_fuel_cost(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
    dt: float = 1.0,
) -> float:
    """Compute true fuel cost for a commitment schedule."""
    N = len(generators)
    schedule = x[: N * T].reshape(N, T)
    return sum(
        gen.c_fuel * gen.P_max * dt * int(schedule[i, t])
        for i, gen in enumerate(generators)
        for t in range(T)
    )


def actual_startup_cost(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
    initial_state: Optional[List[int]] = None,
) -> float:
    """Compute true startup cost for a commitment schedule."""
    if initial_state is None:
        initial_state = [0] * len(generators)
    N = len(generators)
    schedule = x[: N * T].reshape(N, T).astype(int)
    cost = 0.0
    for i, gen in enumerate(generators):
        prev = initial_state[i]
        for t in range(T):
            if schedule[i, t] == 1 and prev == 0:
                cost += gen.SU_cost
            prev = schedule[i, t]
    return cost


def actual_shutdown_cost(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
    initial_state: Optional[List[int]] = None,
) -> float:
    """Compute true shutdown cost for a commitment schedule."""
    if initial_state is None:
        initial_state = [0] * len(generators)
    N = len(generators)
    schedule = x[: N * T].reshape(N, T).astype(int)
    cost = 0.0
    for i, gen in enumerate(generators):
        prev = initial_state[i]
        for t in range(T):
            if schedule[i, t] == 0 and prev == 1:
                cost += gen.SD_cost
            prev = schedule[i, t]
    return cost


def actual_total_cost(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
    dt: float = 1.0,
    initial_state: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute all actual operational costs for a commitment schedule.

    Returns
    -------
    dict with keys 'fuel', 'startup', 'shutdown', 'total'.
    """
    fuel = actual_fuel_cost(x, generators, T, dt)
    startup = actual_startup_cost(x, generators, T, initial_state)
    shutdown = actual_shutdown_cost(x, generators, T, initial_state)
    return {"fuel": fuel, "startup": startup, "shutdown": shutdown, "total": fuel + startup + shutdown}


def power_imbalance(
    x: np.ndarray,
    generators: "List[Generator]",
    demand: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Compute per-timestep power imbalance:  generation − demand.

    Positive = excess generation. Negative = supply shortage.
    """
    decoded = decode_solution(x, generators, T)
    return decoded["total_power"] - np.asarray(demand)


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def solution_report(
    x: np.ndarray,
    generators: "List[Generator]",
    demand: np.ndarray,
    T: int,
    Q: Optional[np.ndarray] = None,
    dt: float = 1.0,
    initial_state: Optional[List[int]] = None,
    title: str = "UNIT COMMITMENT SOLUTION",
) -> str:
    """
    Generate a human-readable solution report.

    Parameters
    ----------
    x : np.ndarray
        Binary solution vector.
    generators, demand, T, dt, initial_state : see other functions.
    Q : np.ndarray, optional
        If provided, QUBO energy is included in the report.
    title : str
        Report title.

    Returns
    -------
    str
    """
    N = len(generators)
    decoded = decode_solution(x, generators, T)
    schedule = decoded["schedule"]
    costs = actual_total_cost(x, generators, T, dt, initial_state)
    imbalance = power_imbalance(x, generators, demand, T)

    W = 62
    sep = "=" * W
    lines: List[str] = []

    lines.append(sep)
    lines.append(f"  FluxQ — {title}")
    lines.append(sep)
    lines.append(f"  System : {N} generators  ×  {T} timesteps  (dt = {dt} h)")
    lines.append(f"  Cost   : ${costs['total']:>12,.2f}")
    lines.append(f"           Fuel     ${costs['fuel']:>10,.2f}")
    lines.append(f"           Startup  ${costs['startup']:>10,.2f}")
    lines.append(f"           Shutdown ${costs['shutdown']:>10,.2f}")

    if Q is not None:
        e = evaluate(Q, x)
        lines.append(f"  QUBO energy : {e:,.4f}")

    lines.append("")
    lines.append("  Commitment schedule  (1 = ON, 0 = OFF)")
    header = f"  {'Generator':<20}" + "  ".join(f"t{t:02d}" for t in range(T))
    lines.append(header)
    lines.append("  " + "-" * (W - 2))
    for i, gen in enumerate(generators):
        row = f"  {gen.name:<20}" + "   ".join(str(schedule[i, t]) for t in range(T))
        lines.append(row)

    lines.append("")
    lines.append(f"  {'t':>3}  {'Demand MW':>10}  {'Gen MW':>8}  {'Imbal MW':>9}  {'Status':>8}")
    lines.append("  " + "-" * (W - 2))
    for t in range(T):
        gen_mw = demand[t] + imbalance[t]
        status = "✓ OK" if abs(imbalance[t]) < 1.0 else f"✗ {imbalance[t]:+.0f}"
        lines.append(
            f"  {t:>3}  {demand[t]:>10.1f}  {gen_mw:>8.1f}  {imbalance[t]:>+9.1f}  {status:>8}"
        )

    n_violations = int(np.sum(np.abs(imbalance) >= 1.0))
    lines.append("")
    lines.append(f"  Power balance violations: {n_violations}/{T} timesteps")
    lines.append(sep)

    return "\n".join(lines)


def check_mut_mdt_feasibility(
    x: np.ndarray,
    generators: "List[Generator]",
    T: int,
    initial_state: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Independently check whether a decoded commitment schedule satisfies
    every generator's MUT/MDT requirements -- by directly walking the
    0/1 schedule itself, with NO reference to any QUBO matrix or penalty
    value.

    Why this exists
    ----------------
    Everywhere else in FluxQ, "does this schedule satisfy MUT/MDT" is
    answered by checking whether the QUBO's MUT/MDT penalty terms
    evaluate to zero (see test_mut_mdt_constraints.py). That's the right
    check when verifying the QUBO FORMULATION itself is correct. But
    when checking whether a SPECIFIC SOLUTION found by a heuristic
    solver is actually feasible -- especially at a scale too large to
    brute-force verify -- checking it with only the same machinery that
    produced it is not independent evidence. This function is a second,
    completely separate method: it never touches Q, only the decoded
    schedule, so agreement between this and the QUBO-based check is
    genuine corroboration, not circular reasoning.

    This function caught a real problem the first time it was used: a
    schedule from an earlier session, previously described as "feasible"
    (based only on exact power balance), was actually MUT-infeasible for
    two generators -- see docs/research-log.md for the full
    investigation, including the surprising finding that the real cause
    was not an under-weighted penalty (fixed, and it didn't change the
    result) but a demand profile that structurally conflicted with full
    MUT compliance no matter how the weights were set.

    Parameters
    ----------
    x : np.ndarray, shape (N*T,) or longer
        Binary solution vector. Only the first N*T entries are used,
        matching every other function in this module -- callers never
        need to slice off any auxiliary MUT/MDT linking variables
        appended after position N*T.
    generators : list of Generator
    T : int
    initial_state : list of int, optional
        Default: all 0 (all generators start OFF).

    Returns
    -------
    dict with keys:
        'feasible' : bool -- True iff zero violations were found.
        'violations' : list of dict, one per violation, each with keys
            'generator', 'type' ('MUT' or 'MDT'), 'start_t', and a
            human-readable 'message'.

    Notes on the horizon boundary
    ------------------------------
    A run of ON or OFF timesteps that reaches all the way to the end of
    the visible horizon is handled two different ways depending on
    whether it's already long enough:
      - If the observed run length already meets MUT/MDT, it's
        confirmed satisfied -- the fact that it *also* continues past
        what's visible doesn't undo that.
      - If the observed run length falls short AND it reaches the end
        of the horizon (rather than ending via a genuine transition
        back), it's treated as inconclusive, not a violation -- matching
        the same "can't penalize what's beyond the horizon" convention
        used throughout constraints.py.
      - If the run falls short AND it ends via a genuine, fully-observed
        transition back, that's a real, confirmed violation.
    An earlier draft of this function conflated the first and second
    cases (checking only whether the run ended within the horizon,
    before checking whether it was already long enough), which produced
    the right final feasible/infeasible answer on every hand-derived
    test case by coincidence, but for the wrong reason -- fixed before
    being trusted as an independent check. See docs/research-log.md.
    """
    if initial_state is None:
        initial_state = [0] * len(generators)

    N = len(generators)
    schedule = x[: N * T].reshape(N, T).astype(int)

    violations: List[Dict[str, Any]] = []

    for i, gen in enumerate(generators):
        full_seq = [initial_state[i]] + list(schedule[i])

        t = 1
        while t < len(full_seq):
            if full_seq[t] == full_seq[t - 1]:
                t += 1
                continue

            new_state = full_seq[t]
            run_start_idx = t - 1

            run_len = 1
            tt = t + 1
            while tt < len(full_seq) and full_seq[tt] == new_state:
                run_len += 1
                tt += 1
            hit_boundary = (tt == len(full_seq))

            min_required = gen.MUT if new_state == 1 else gen.MDT

            if run_len < min_required and not hit_boundary:
                kind = "MUT" if new_state == 1 else "MDT"
                state_word = "on" if new_state == 1 else "off"
                violations.append({
                    "generator": gen.name,
                    "type": kind,
                    "start_t": run_start_idx,
                    "message": (
                        f"{gen.name}: {'started' if new_state == 1 else 'shut down'} "
                        f"at t={run_start_idx}, stayed {state_word} for only "
                        f"{run_len} timestep(s), needs {min_required} ({kind})."
                    ),
                })

            t += 1

    return {"feasible": len(violations) == 0, "violations": violations}