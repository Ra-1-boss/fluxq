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
    λ₁  ≥  max(c_i · P_i · dt · T + SU_i · T) / min(P_i)²
    (this is the exact bound computed by QUBOBuilder.suggest_lambda() —
    it includes worst-case startup cost, which a simplified fuel-only
    estimate would omit.)

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
# Phase 1 — minimum up/down time (implemented) and ramp rate (still a stub)
# ---------------------------------------------------------------------------
#
# MUT/MDT are genuinely harder than power balance or spinning reserve: they
# involve THREE binary variables at once (x_{t-1}, x_t, x_{t+k}), which
# expands to a cubic term that QUBO -- strictly pairwise by definition --
# cannot represent directly. This is exactly the gap identified reading
# Glover, Kochenberger & Du (2019): every worked example in that tutorial
# is a single-timestep constraint; none of them couple a variable to itself
# across multiple future timesteps the way MUT/MDT require.
#
# The fix is Rosenberg's classical cubic-to-quadratic reduction (Rosenberg,
# 1975 -- referenced in passing in Glover et al. section 7, without a
# worked example). Introduce one auxiliary binary variable per (generator,
# timestep) pair, w_{i,t}, meant to equal the product x_{i,t-1}*x_{i,t}.
# Force it to that value with a penalty, then substitute w_{i,t} everywhere
# that product would otherwise appear -- collapsing every cubic term back
# down to degree <= 2.
#
# MUT and MDT both need this same product (x_{t-1}*x_t appears in both the
# startup indicator and the shutdown indicator), so they share one w block
# and one linking penalty rather than needing two.


def add_transition_linking(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    n_vars_real: int,
    lambda_link: float,
) -> None:
    """
    Force each auxiliary variable w_{i,t} to equal x_{i,t-1} * x_{i,t}.

    w_{i,t} is not a real commitment decision -- it exists purely so
    MUT/MDT constraints can be expressed in QUBO's strictly pairwise form.
    Without it, expanding the startup indicator
    (x_{i,t} - x_{i,t}*x_{i,t-1}) against a future timestep x_{i,t+k}
    produces a genuine cubic term x_{i,t}*x_{i,t-1}*x_{i,t+k}, which a
    QUBO matrix cannot represent directly.

    Penalty (Rosenberg's substitution penalty, verified by hand against
    all 8 combinations of a, b, w -- see docs/research-log.md):

        P_link(a, b, w) = a*b - 2*a*w - 2*b*w + 3*w

    This equals exactly 0 when w = a*b, and a strictly positive integer
    (1 or 3) for every other combination of a, b, w -- confirmed by
    direct enumeration before this was written as code.

    w_{i,t} indices run for t in [1, T) -- there is no w_{i,0}; the t=0
    boundary is handled directly in terms of the real x_{i,0} variable in
    add_mut_constraint()/add_mdt_constraint(), matching the existing SU/SD
    boundary treatment already used in objective.py.

    Note on efficiency: this links every generator's w block unconditionally,
    even for generators whose MUT and MDT are both 1 (no real constraint,
    e.g. several of the real RTS-GMLC oil/gas peakers). Those w's end up
    unused but still cost matrix entries. Worth optimizing later if it
    matters at scale; correctness comes first here.

    Parameters
    ----------
    Q : np.ndarray, shape (n_vars_real + N*(T-1), same)
        QUBO matrix, already sized to include the auxiliary w block.
    generators : list of Generator
    T : int
    n_vars_real : int
        N * T -- the offset in Q where the auxiliary w block begins.
    lambda_link : float
        Penalty weight. Must be large enough that the optimizer can never
        find it cheaper to let w disagree with the true product in order
        to dodge a MUT/MDT penalty. See
        ``QUBOBuilder.suggest_lambda_link()`` for an automatic estimate.
    """
    N = len(generators)
    for i in range(N):
        for t in range(1, T):
            a = i * T + (t - 1)                         # idx(i, t-1)
            b = i * T + t                                # idx(i, t)
            w = n_vars_real + i * (T - 1) + (t - 1)      # idx of w_{i,t}
            # a < b < n_vars_real <= w always holds here, so every term
            # below is already written with the smaller index first --
            # required to keep Q strictly upper-triangular.
            Q[a, b] += lambda_link * 1.0
            Q[a, w] += lambda_link * (-2.0)
            Q[b, w] += lambda_link * (-2.0)
            Q[w, w] += lambda_link * 3.0


def add_mut_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    n_vars_real: int,
    lambda_3: float,
    initial_state: Optional[List[int]] = None,
) -> None:
    """
    Minimum Up Time (MUT) constraint.

    If generator i starts up at timestep t (transitions OFF -> ON), it
    must remain ON for at least MUT_i consecutive timesteps: x_{i,t+k} = 1
    for k = 1, ..., MUT_i - 1.

    Derivation
    ----------
    started_{i,t} = x_{i,t} - x_{i,t}*x_{i,t-1}     (t >= 1)
                  = x_{i,t} - w_{i,t}                (using the linking
                                                        substitution)

    violation_{i,t,k} = started_{i,t} * (1 - x_{i,t+k})
                       = x_{i,t} - x_{i,t}*x_{i,t+k} - w_{i,t}
                         + w_{i,t}*x_{i,t+k}

    Every term is now degree <= 2. Without the substitution, expanding
    started_{i,t} in terms of real x's alone leaves a genuine cubic term
    (x_{i,t}*x_{i,t-1}*x_{i,t+k}) that QUBO cannot represent.

    The t=0 boundary has no real x_{i,-1} variable, so it's handled
    directly -- matching the SU-cost boundary treatment already in
    objective.py: if initial_state[i] == 0, started_{i,0} = x_{i,0}
    (no w needed), giving violation_{i,0,k} = x_{i,0} - x_{i,0}*x_{i,k}.

    Only violations fully inside the planning horizon (t+k < T) are
    penalized. A start event too close to the end of the horizon to
    fully check is not flagged -- a deliberate, documented modeling
    simplification, not an oversight.

    Requires add_transition_linking() to already have been called on the
    same Q. This function assumes w_{i,t} is being correctly forced to
    x_{i,t-1}*x_{i,t} elsewhere; it does not add that penalty itself,
    since MUT and MDT share the same w block and the link penalty should
    only be added once regardless of how many of them are active.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix, already sized for N*T + N*(T-1) variables.
    generators : list of Generator
    T : int
    n_vars_real : int
        N * T -- offset in Q where the w block begins.
    lambda_3 : float
        Penalty weight for MUT violations.
    initial_state : list of int, optional
        Default: all 0 (all generators start OFF), matching the rest of
        FluxQ's boundary convention.
    """
    if initial_state is None:
        initial_state = [0] * len(generators)

    for i, gen in enumerate(generators):
        mut = gen.MUT
        if mut <= 1:
            continue   # MUT <= 1 means no constraint beyond binary on/off

        # t = 0 boundary -- no w needed, matches objective.py's SU pattern.
        if initial_state[i] == 0:
            idx_0 = i * T
            for k in range(1, mut):
                if k >= T:
                    break
                idx_k = i * T + k
                # violation = x_0 - x_0*x_k
                Q[idx_0, idx_0] += lambda_3 * 1.0
                Q[idx_0, idx_k] += lambda_3 * (-1.0)

        # t >= 1 -- uses the linked w_{i,t}.
        for t in range(1, T):
            w = n_vars_real + i * (T - 1) + (t - 1)
            idx_t = i * T + t
            for k in range(1, mut):
                if t + k >= T:
                    break
                idx_tk = i * T + t + k
                # violation = x_t - x_t*x_{t+k} - w + w*x_{t+k}
                Q[idx_t, idx_t] += lambda_3 * 1.0
                Q[idx_t, idx_tk] += lambda_3 * (-1.0)
                Q[w, w] += lambda_3 * (-1.0)
                # idx_tk < w always (idx_tk < n_vars_real <= w) -- smaller
                # index first, keeping Q strictly upper-triangular.
                Q[idx_tk, w] += lambda_3 * 1.0


def add_mdt_constraint(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    n_vars_real: int,
    lambda_4: float,
    initial_state: Optional[List[int]] = None,
) -> None:
    """
    Minimum Down Time (MDT) constraint.

    Symmetric to MUT: if generator i shuts down at timestep t (transitions
    ON -> OFF), it must remain OFF for at least MDT_i consecutive
    timesteps: x_{i,t+k} = 0 for k = 1, ..., MDT_i - 1.

    Derivation
    ----------
    shut_{i,t} = x_{i,t-1} - x_{i,t-1}*x_{i,t}      (t >= 1)
               = x_{i,t-1} - w_{i,t}                 (linking substitution)

    violation_{i,t,k} = shut_{i,t} * x_{i,t+k}
                       = x_{i,t-1}*x_{i,t+k} - w_{i,t}*x_{i,t+k}

    Same reduction as MUT: expanding shut_{i,t} fully in terms of real x's
    leaves a cubic term (x_{i,t-1}*x_{i,t}*x_{i,t+k}); substituting w_{i,t}
    removes it, reusing the same w block MUT already relies on -- no
    second auxiliary variable needed.

    The t=0 boundary applies only when initial_state[i] == 1 (the
    generator was already ON before the horizon began): shut_{i,0} =
    1 - x_{i,0}, giving violation_{i,0,k} = x_{i,k} - x_{i,0}*x_{i,k}
    (constant dropped, matching the SD-cost boundary treatment already in
    objective.py).

    Requires add_transition_linking() to already have been called on the
    same Q -- see that function's docstring for why.

    Parameters
    ----------
    Q : np.ndarray
    generators : list of Generator
    T : int
    n_vars_real : int
        N * T -- offset in Q where the w block begins.
    lambda_4 : float
        Penalty weight for MDT violations.
    initial_state : list of int, optional
        Default: all 0.
    """
    if initial_state is None:
        initial_state = [0] * len(generators)

    for i, gen in enumerate(generators):
        mdt = gen.MDT
        if mdt <= 1:
            continue

        # t = 0 boundary -- only relevant if the generator was ON before
        # the horizon began (otherwise there's no shutdown event at t=0).
        if initial_state[i] == 1:
            idx_0 = i * T
            for k in range(1, mdt):
                if k >= T:
                    break
                idx_k = i * T + k
                # violation = x_k - x_0*x_k
                Q[idx_k, idx_k] += lambda_4 * 1.0
                Q[idx_0, idx_k] += lambda_4 * (-1.0)

        # t >= 1 -- uses the same linked w_{i,t} as MUT.
        for t in range(1, T):
            w = n_vars_real + i * (T - 1) + (t - 1)
            idx_tm1 = i * T + (t - 1)
            for k in range(1, mdt):
                if t + k >= T:
                    break
                idx_tk = i * T + t + k
                # violation = x_{t-1}*x_{t+k} - w*x_{t+k}
                Q[idx_tm1, idx_tk] += lambda_4 * 1.0
                # idx_tk < w always -- smaller index first.
                Q[idx_tk, w] += lambda_4 * (-1.0)


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
