"""
Objective function terms for the FluxQ QUBO formulation.

All functions modify the QUBO matrix Q **in-place** using the
upper-triangular convention:

  E(x) = x @ Q @ x  (valid for upper-triangular Q and binary x ∈ {0,1}^n)

Because x_i² = x_i for binary variables, linear terms land on the diagonal
and quadratic cross-terms land in the strict upper triangle.

Encoding rules
--------------
Linear term  ``a · x_i``:
    Q[i, i] += a

Quadratic term  ``a · x_i · x_j``  with  idx(i) < idx(j):
    Q[idx(i), idx(j)] += a

These two rules are the *only* ways Q is modified in this module.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.grid_builder import Generator


# ---------------------------------------------------------------------------
# Fuel cost
# ---------------------------------------------------------------------------

def add_fuel_cost(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    dt: float = 1.0,
) -> None:
    """
    Add fuel (variable) cost terms to Q.

    .. math::

        \\text{Fuel} = \\sum_{i,t} c_i \\cdot P_i \\cdot x_{i,t} \\cdot \\Delta t

    This is linear in x → diagonal entries only.

    Parameters
    ----------
    Q : np.ndarray, shape (N·T, N·T)
        QUBO matrix (upper-triangular, modified in-place).
    generators : list of Generator
        Each generator supplies ``c_fuel`` ($/MWh) and ``P_max`` (MW).
    T : int
        Number of timesteps.
    dt : float
        Timestep duration in hours. Default 1.0.
    """
    for i, gen in enumerate(generators):
        for t in range(T):
            idx = i * T + t
            Q[idx, idx] += gen.c_fuel * gen.P_max * dt


# ---------------------------------------------------------------------------
# Startup cost
# ---------------------------------------------------------------------------

def add_startup_cost(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    initial_state: Optional[List[int]] = None,
) -> None:
    """
    Add startup cost terms to Q.

    A startup occurs at timestep t when a generator transitions OFF → ON.
    For binary variables, the indicator equals:

    .. math::

        \\text{start}_{i,t} = \\max(0,\\, x_{i,t} - x_{i,t-1})
                             = x_{i,t} - x_{i,t} \\cdot x_{i,t-1}

    So the startup cost decomposes into:

    .. math::

        \\text{SU}_i \\cdot x_{i,t}
        - \\text{SU}_i \\cdot x_{i,t-1} \\cdot x_{i,t}

    QUBO contributions
    ~~~~~~~~~~~~~~~~~~
    For t ≥ 1  (let p = idx(i, t-1), q = idx(i, t), p < q):
      - ``Q[q, q] += SU_i``        (linear term on x_{i,t})
      - ``Q[p, q] -= SU_i``        (quadratic cross-term)

    For t = 0 with initial_state[i] = 0  (generator was OFF):
      - ``Q[idx(i,0), idx(i,0)] += SU_i``

    Correctness check (t ≥ 1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    =========  =========  ===========  =============
    x_{t-1}    x_t        Actual cost  QUBO energy*
    =========  =========  ===========  =============
    0          0          0            0
    0          1          SU_i         SU_i
    1          0          0            0
    1          1          0            SU_i − SU_i = 0
    =========  =========  ===========  =============
    *Showing only the startup-related portion of the diagonal and off-diagonal.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (modified in-place).
    generators : list of Generator
    T : int
    initial_state : list of int, optional
        x_{i,-1} for each generator. Default: all 0 (all generators start OFF).
    """
    if initial_state is None:
        initial_state = [0] * len(generators)

    for i, gen in enumerate(generators):
        # t = 0 boundary condition
        if initial_state[i] == 0:
            # Generator was OFF → startup cost if x_{i,0} = 1
            idx_0 = i * T
            Q[idx_0, idx_0] += gen.SU_cost
        # If initial_state[i] == 1 the generator was already running; no startup at t=0.

        # t ≥ 1
        for t in range(1, T):
            p = i * T + (t - 1)   # idx(i, t-1) — always less than q
            q = i * T + t          # idx(i, t)
            Q[q, q] += gen.SU_cost           # linear: +SU on x_{i,t}
            Q[p, q] -= gen.SU_cost           # quadratic: −SU on x_{i,t-1}·x_{i,t}


# ---------------------------------------------------------------------------
# Shutdown cost
# ---------------------------------------------------------------------------

def add_shutdown_cost(
    Q: np.ndarray,
    generators: "List[Generator]",
    T: int,
    initial_state: Optional[List[int]] = None,
) -> None:
    """
    Add shutdown cost terms to Q.

    A shutdown occurs at timestep t when a generator transitions ON → OFF.
    For binary variables:

    .. math::

        \\text{shut}_{i,t} = \\max(0,\\, x_{i,t-1} - x_{i,t})
                            = x_{i,t-1} - x_{i,t-1} \\cdot x_{i,t}

    Shutdown cost decomposes as:

    .. math::

        \\text{SD}_i \\cdot x_{i,t-1}
        - \\text{SD}_i \\cdot x_{i,t-1} \\cdot x_{i,t}

    QUBO contributions
    ~~~~~~~~~~~~~~~~~~
    For t ≥ 1  (p = idx(i, t-1), q = idx(i, t)):
      - ``Q[p, p] += SD_i``        (linear term on x_{i,t-1})
      - ``Q[p, q] -= SD_i``        (quadratic cross-term)

    For t = 0 with initial_state[i] = 1  (generator was ON):
      ``SD_i · (1 − x_{i,0}) = SD_i − SD_i · x_{i,0}``
      → ``Q[idx(i,0), idx(i,0)] -= SD_i``
      (The constant SD_i is dropped as it does not affect the argmin.)

    Combined effect of startup + shutdown on off-diagonal entry Q[p, q]
    (same generator, adjacent timesteps):
      ``Q[p, q] -= (SU_i + SD_i)``

    Correctness check (t ≥ 1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    =========  =========  ===========  =============
    x_{t-1}    x_t        Actual cost  QUBO energy*
    =========  =========  ===========  =============
    0          0          0            0
    0          1          0            0
    1          0          SD_i         SD_i
    1          1          0            SD_i − SD_i = 0
    =========  =========  ===========  =============

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (modified in-place).
    generators : list of Generator
    T : int
    initial_state : list of int, optional
        Default: all 0.
    """
    if initial_state is None:
        initial_state = [0] * len(generators)

    for i, gen in enumerate(generators):
        # t = 0 boundary condition
        if initial_state[i] == 1:
            # Generator was ON → shutdown if x_{i,0} = 0
            # Encodes: SD_i · (1 - x_{i,0}) → drop constant, add -SD_i to diagonal
            idx_0 = i * T
            Q[idx_0, idx_0] -= gen.SD_cost

        # t ≥ 1
        for t in range(1, T):
            p = i * T + (t - 1)
            q = i * T + t
            Q[p, p] += gen.SD_cost           # linear: +SD on x_{i,t-1}
            Q[p, q] -= gen.SD_cost           # quadratic: −SD on x_{i,t-1}·x_{i,t}
