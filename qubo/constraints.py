"""
Constraint penalty terms for the FluxQ QUBO formulation.

Constraints are encoded as squared-penalty functions added to the objective.
A violated constraint raises the QUBO energy by a large amount, discouraging
the optimizer from choosing infeasible solutions.

Penalty architecture
--------------------
Each constraint C has a target value (or range) and a penalty weight λ.
For equality constraint  Σ f(x) = target:
    Penalty = λ · (Σ f(x) − target)²

Expanding the square decomposes into diagonal (linear-in-x) and off-diagonal
(quadratic-in-x) QUBO terms, using x_i² = x_i for binary variables.

Phase coverage
--------------
Phase 0 (this file):   Power balance, spinning reserve (equality approx.)
Phase 1 (TODO):        Minimum up/down time, ramp rate
Phase 2 (TODO):        Renewable-scenario extensions
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.grid_builder import Generator


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _add_squared_linear_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    targets: np.ndarray,
    lam: float,
) -> None:
    """
    Add penalty for a per-timestep constraint of the form:

        Σ_i  P_i · x_{i,t}  =  target_t     for all t

    Expanding  (Σ_i P_i x_{i,t} − target_t)²:

        = Σ_i P_i² x_i  +  2 Σ_{i<j} P_i P_j x_i x_j  −  2 D_t Σ_i P_i x_i  +  D_t²

    Constants (D_t²) are dropped — they do not affect the argmin.

    QUBO contributions
    ~~~~~~~~~~~~~~~~~~
    Diagonal  (i, t):   ``lam * P_i * (P_i − 2 * target_t)``
    Off-diag  (i<j, same t): ``lam * 2 * P_i * P_j``
    """
    N = len(generators)
    P = np.array([gen.P_max for gen in generators], dtype=float)

    for t in range(T):
        tgt = targets[t]

        # Diagonal terms
        for i in range(N):
            idx_i = i * T + t
            Q[idx_i, idx_i] += lam * P[i] * (P[i] - 2.0 * tgt)

        # Off-diagonal terms (same timestep, i < j guaranteed since i < j → i·T+t < j·T+t)
        for i in range(N):
            for j in range(i + 1, N):
                idx_i = i * T + t
                idx_j = j * T + t
                Q[idx_i, idx_j] += lam * 2.0 * P[i] * P[j]


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------

def add_power_balance(
    Q: np.ndarray,
    generators: "List[Generator]",
    demand: np.ndarray,
    T: int,
    lambda_1: float,
) -> None:
    """
    Add power balance (equality) penalty to Q.

    Constraint
    ----------
    .. math::

        \\sum_i P_i \\cdot x_{i,t} = D_t \\quad \\forall\\, t

    Penalty
    -------
    .. math::

        \\lambda_1 \\sum_t \\left(\\sum_i P_i \\cdot x_{i,t} - D_t\\right)^2

    Choosing λ₁
    ~~~~~~~~~~~
    λ₁ must be large enough that satisfying the constraint is always preferred
    over paying cheaper fuel costs. A safe lower bound is::

        λ₁  ≥  max(c_i · P_i · dt) / min(P_i)²  ×  T

    Use ``QUBOBuilder.suggest_lambda()`` for an automatic estimate.

    Parameters
    ----------
    Q : np.ndarray, shape (N·T, N·T)
        QUBO matrix (modified in-place, upper-triangular).
    generators : list of Generator
    demand : np.ndarray, shape (T,)
        Electricity demand D_t (MW) at each timestep.
    T : int
    lambda_1 : float
        Penalty weight for power balance.
    """
    demand = np.asarray(demand, dtype=float)
    _add_squared_linear_constraint(Q, generators, T, targets=demand, lam=lambda_1)


# ---------------------------------------------------------------------------
# Spinning reserve
# ---------------------------------------------------------------------------

def add_spinning_reserve(
    Q: np.ndarray,
    generators: "List[Generator]",
    demand: np.ndarray,
    reserve: np.ndarray,
    T: int,
    lambda_2: float,
) -> None:
    """
    Add spinning reserve penalty to Q.

    Constraint (inequality)
    -----------------------
    .. math::

        \\sum_i P_i \\cdot x_{i,t} \\geq D_t + R_t \\quad \\forall\\, t

    Phase 0 approximation
    ~~~~~~~~~~~~~~~~~~~~~
    We encode this as an *equality* penalty:

    .. math::

        \\lambda_2 \\sum_t \\left(\\sum_i P_i \\cdot x_{i,t} - (D_t + R_t)\\right)^2

    This penalises both under-reserve *and* excess capacity online. The
    approximation is acceptable when λ₂ < λ₁ (power balance dominates) and
    the demand profile is chosen such that reserve is typically binding.

    **Phase 1 improvement**: encode with slack binary variables to get a
    true one-sided inequality penalty.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (modified in-place).
    generators : list of Generator
    demand : np.ndarray, shape (T,)
    reserve : np.ndarray, shape (T,)
        Required spinning reserve R_t (MW) at each timestep.
    T : int
    lambda_2 : float
        Penalty weight for reserve constraint. Typically λ₂ ≤ λ₁/2.
    """
    demand = np.asarray(demand, dtype=float)
    reserve = np.asarray(reserve, dtype=float)
    targets = demand + reserve
    _add_squared_linear_constraint(Q, generators, T, targets=targets, lam=lambda_2)


# ---------------------------------------------------------------------------
# Phase 1 stubs — minimum up/down time and ramp rate
# ---------------------------------------------------------------------------

def add_mut_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    lambda_3: float,
) -> None:
    """
    [Phase 1] Minimum Up Time (MUT) constraint.

    If generator i starts at time t, it must remain ON for at least MUT_i
    consecutive timesteps.

    Encoding uses auxiliary startup indicator variables y_{i,t}:
        y_{i,t} = max(0, x_{i,t} − x_{i,t−1})

    Penalty:
        λ₃ · Σ_{i,t} y_{i,t} · Σ_{k=1}^{MUT_i−1} (1 − x_{i,t+k})²

    This encoding introduces N·T additional binary variables, doubling the
    problem size. See docs/formulation.md for the full derivation.

    Status: NOT YET IMPLEMENTED — Phase 1.
    """
    raise NotImplementedError(
        "MUT constraint encoding is a Phase 1 feature. "
        "See docs/formulation.md for the derivation plan."
    )


def add_mdt_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    lambda_4: float,
) -> None:
    """
    [Phase 1] Minimum Down Time (MDT) constraint.

    Symmetric to MUT: if generator i shuts down at time t, it must remain
    OFF for at least MDT_i consecutive timesteps.

    Status: NOT YET IMPLEMENTED — Phase 1.
    """
    raise NotImplementedError(
        "MDT constraint encoding is a Phase 1 feature. "
        "See docs/formulation.md for the derivation plan."
    )


def add_ramp_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    lambda_5: float,
) -> None:
    """
    [Phase 1] Ramp rate constraint.

    Generator output cannot change faster than RR_i MW/timestep.
    In the binary model (on at P_max or off at 0), this becomes a
    constraint on consecutive commitment decisions.

    Penalty:
        λ₅ · Σ_{i,t} max(0, |P_i · (x_{i,t} − x_{i,t−1})| − RR_i)²

    Status: NOT YET IMPLEMENTED — Phase 1.
    """
    raise NotImplementedError(
        "Ramp rate constraint encoding is a Phase 1 feature. "
        "See docs/formulation.md for the derivation plan."
    )
