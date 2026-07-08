"""
Core QUBO builder for the FluxQ Unit Commitment framework.

QUBOBuilder is the main entry point. It assembles the full QUBO matrix Q
from objective terms and constraint penalties, and provides helpers for
evaluation, decoding, and diagnostics.

Variable convention
-------------------
  x_{i,t} ∈ {0,1}  where  i = generator index, t = timestep index.
  QUBO index:  idx(i, t) = i * T + t   (row-major, generators as rows)
  For N generators and T timesteps:  n = N * T total binary variables.

QUBO energy convention
----------------------
  E(x) = x^T Q x  with Q upper-triangular.
  Linear terms  a·x_i  →  Q[i, i] += a
  Quadratic terms  a·x_i·x_j  (i < j)  →  Q[i, j] += a

  NumPy: ``energy = float(x @ Q @ x)``

Phase coverage
--------------
  Phase 0 (implemented):  objective (fuel, startup, shutdown) + power balance
  Phase 1 (stubs):        MUT, MDT, ramp rate constraints
  Phase 2 (planned):      renewable scenario extensions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.grid_builder import Generator
from .objective import add_fuel_cost, add_startup_cost, add_shutdown_cost
from .constraints import add_power_balance, add_spinning_reserve
from .validator import (
    evaluate,
    decode_solution,
    solution_report,
    check_qubo_matrix,
    actual_total_cost,
    power_imbalance,
)


# ---------------------------------------------------------------------------
# Penalty weight configuration
# ---------------------------------------------------------------------------

@dataclass
class LambdaConfig:
    """
    Penalty weights for each QUBO constraint term.

    Each λ balances constraint satisfaction against cost minimisation.
    If λ is too small, constraints will be violated in the optimal solution.
    If λ is too large, the optimizer ignores cost and just satisfies constraints.

    The right balance depends on the scale of the objective function:

    .. code-block:: text

        λ₁ (power balance)  ≥  max(c_i · P_i · dt · T + SU_i · T) / min(P_i)²
        (includes worst-case startup cost — see suggest_lambda() for the exact formula)

    Use ``QUBOBuilder.suggest_lambda()`` for an automatic lower-bound estimate.

    Attributes
    ----------
    lambda_1 : float
        Power balance penalty weight. Most important — must be set correctly.
    lambda_2 : float
        Spinning reserve penalty weight. Typically λ₂ ≤ λ₁ / 2.
    lambda_3 : float
        Minimum up time penalty weight. (Phase 1)
    lambda_4 : float
        Minimum down time penalty weight. (Phase 1)
    lambda_5 : float
        Ramp rate penalty weight. (Phase 1)
    """
    lambda_1: float = 1.0    # power balance
    lambda_2: float = 0.5    # spinning reserve
    lambda_3: float = 1.0    # MUT (Phase 1)
    lambda_4: float = 1.0    # MDT (Phase 1)
    lambda_5: float = 0.5    # ramp rate (Phase 1)

    def __post_init__(self) -> None:
        for name in ("lambda_1", "lambda_2", "lambda_3", "lambda_4", "lambda_5"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"LambdaConfig.{name} must be ≥ 0, got {v}")


# ---------------------------------------------------------------------------
# Main QUBO builder
# ---------------------------------------------------------------------------

class QUBOBuilder:
    """
    Assemble the QUBO matrix for the Unit Commitment Problem.

    Parameters
    ----------
    generators : list of Generator
        Generator objects, ordered by index i = 0, 1, …, N−1.
    T : int
        Number of timesteps in the planning horizon.
    dt : float
        Duration of each timestep in hours. Default 1.0.
    initial_state : list of int, optional
        x_{i, −1} — commitment state at the timestep *before* the horizon.
        0 = OFF, 1 = ON. Default: all generators start OFF.

    Examples
    --------
    >>> from fluxq.data.grid_builder import make_toy_generators
    >>> import numpy as np
    >>> gens = make_toy_generators()
    >>> builder = QUBOBuilder(gens, T=4)
    >>> demand = np.array([200.0, 280.0, 350.0, 220.0])
    >>> Q = builder.build(demand, LambdaConfig(lambda_1=0.01))
    >>> Q.shape
    (12, 12)
    >>> builder.n_vars
    12
    """

    def __init__(
        self,
        generators: List[Generator],
        T: int,
        dt: float = 1.0,
        initial_state: Optional[List[int]] = None,
    ) -> None:
        if len(generators) == 0:
            raise ValueError("At least one generator is required.")
        if T < 1:
            raise ValueError(f"T must be ≥ 1, got {T}.")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        self.generators = list(generators)
        self.N: int = len(generators)
        self.T: int = T
        self.dt: float = dt
        self.n_vars: int = self.N * self.T
        self.initial_state: List[int] = (
            list(initial_state) if initial_state is not None else [0] * self.N
        )

        if len(self.initial_state) != self.N:
            raise ValueError(
                f"initial_state must have length N={self.N}, "
                f"got {len(self.initial_state)}."
            )
        # NOTE: value validation (0/1 check) is Step 3 — not yet added here.

        self._Q: Optional[np.ndarray] = None
        self._demand: Optional[np.ndarray] = None
        self._lambdas: Optional[LambdaConfig] = None

    # ------------------------------------------------------------------
    # Variable index helpers
    # ------------------------------------------------------------------

    def var_idx(self, i: int, t: int) -> int:
        """
        QUBO index for generator i at timestep t.

        Index = i * T + t  (row-major: generators as rows, time as columns).

        Parameters
        ----------
        i : int  — generator index in [0, N)
        t : int  — timestep index in [0, T)
        """
        if not (0 <= i < self.N):
            raise IndexError(f"Generator index {i} out of range [0, {self.N - 1}].")
        if not (0 <= t < self.T):
            raise IndexError(f"Timestep {t} out of range [0, {self.T - 1}].")
        return i * self.T + t

    def idx_to_gen_time(self, idx: int):
        """Inverse of var_idx: return (i, t) for a given QUBO index."""
        if not (0 <= idx < self.n_vars):
            raise IndexError(f"QUBO index {idx} out of range [0, {self.n_vars - 1}].")
        return divmod(idx, self.T)   # (i, t)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        demand: np.ndarray,
        lambdas: Optional[LambdaConfig] = None,
        reserve: Optional[np.ndarray] = None,
        include_reserve: bool = False,
    ) -> np.ndarray:
        """
        Build the full QUBO matrix Q.

        Phase 0 contents
        ----------------
        1. Objective: fuel cost, startup cost, shutdown cost.
        2. Power balance penalty (equality, all timesteps).
        3. [Optional] Spinning reserve penalty.

        Parameters
        ----------
        demand : np.ndarray, shape (T,)
            Electricity demand D_t (MW) at each timestep.
        lambdas : LambdaConfig, optional
            Penalty weights. Default: LambdaConfig() (all weights = 1.0).
        reserve : np.ndarray, shape (T,), optional
            Required spinning reserve R_t (MW). If None and
            ``include_reserve=True``, defaults to 10% of demand.
        include_reserve : bool
            Whether to add the spinning reserve penalty. Default False.

        Returns
        -------
        np.ndarray, shape (N·T, N·T)
            Upper-triangular QUBO matrix Q.
            Energy: ``E(x) = float(x @ Q @ x)``.
        """
        demand = np.asarray(demand, dtype=float)
        if demand.shape != (self.T,):
            raise ValueError(
                f"demand must have shape ({self.T},), got {demand.shape}."
            )
        if np.any(demand < 0):
            raise ValueError("demand values must be non-negative.")

        if lambdas is None:
            lambdas = LambdaConfig()

        # Fresh upper-triangular matrix
        Q = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)

        # ── Objective terms ──────────────────────────────────────────────
        add_fuel_cost(Q, self.generators, self.T, self.dt)
        add_startup_cost(Q, self.generators, self.T, self.initial_state)
        add_shutdown_cost(Q, self.generators, self.T, self.initial_state)

        # ── Constraint penalties ─────────────────────────────────────────
        add_power_balance(Q, self.generators, demand, self.T, lambdas.lambda_1)

        if include_reserve:
            if reserve is None:
                reserve = demand * 0.10
            reserve = np.asarray(reserve, dtype=float)
            if reserve.shape != (self.T,):
                raise ValueError(
                    f"reserve must have shape ({self.T},), got {reserve.shape}."
                )
            add_spinning_reserve(
                Q, self.generators, demand, reserve, self.T, lambdas.lambda_2
            )

        self._Q = Q
        self._demand = demand.copy()
        self._lambdas = lambdas
        return Q

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        """The built QUBO matrix. Raises RuntimeError if build() hasn't been called."""
        if self._Q is None:
            raise RuntimeError("Call build() before accessing Q.")
        return self._Q

    @property
    def demand(self) -> np.ndarray:
        """The demand array used in the last build() call."""
        if self._demand is None:
            raise RuntimeError("Call build() first.")
        return self._demand

    # ------------------------------------------------------------------
    # Evaluation and decoding
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """Compute QUBO energy E(x) = x^T Q x."""
        return evaluate(self.Q, x)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode binary vector x into a commitment schedule dict."""
        return decode_solution(x, self.generators, self.T)

    def costs(
        self,
        x: np.ndarray,
        initial_state: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute actual operational costs for solution x.

        These are the *true* costs, computed directly from the schedule —
        not derived from the QUBO energy (which includes penalty terms and
        dropped constants).
        """
        return actual_total_cost(
            x, self.generators, self.T, self.dt,
            initial_state or self.initial_state,
        )

    def imbalance(self, x: np.ndarray) -> np.ndarray:
        """Per-timestep power imbalance (generation − demand) for solution x."""
        return power_imbalance(x, self.generators, self.demand, self.T)

    def report(self, x: np.ndarray, title: str = "UNIT COMMITMENT SOLUTION") -> str:
        """Generate a human-readable solution report."""
        return solution_report(
            x, self.generators, self.demand, self.T,
            Q=self.Q, dt=self.dt, initial_state=self.initial_state,
            title=title,
        )

    # ------------------------------------------------------------------
    # Diagnostics and metadata
    # ------------------------------------------------------------------

    def suggest_lambda(self) -> float:
        """
        Estimate a safe lower bound for λ₁ (power balance penalty weight).

        Derivation
        ----------
        We want:  λ₁ · min(P_i)²  >  max(c_i · P_i · dt · T)

        So that the cost of a 1-unit power imbalance (in P_min² units)
        always exceeds the maximum possible objective savings. This ensures
        the power balance constraint is never violated to save cost.

        Returns
        -------
        float
            Suggested minimum value for LambdaConfig.lambda_1.
        """
        max_obj = max(
            gen.c_fuel * gen.P_max * self.dt * self.T
            + gen.SU_cost * self.T          # worst case: start every timestep
            for gen in self.generators
        )
        min_P_sq = min(gen.P_max for gen in self.generators) ** 2
        return max_obj / min_P_sq

    def validate(self) -> Dict[str, Any]:
        """Run structural validation on the built QUBO matrix."""
        return check_qubo_matrix(self.Q, self.n_vars)

    def info(self) -> Dict[str, Any]:
        """
        Return metadata about the QUBO problem.

        Useful for scaling analysis (Phase 4 — QUBO size vs grid size).
        """
        n = self.n_vars
        return {
            "n_generators": self.N,
            "n_timesteps": self.T,
            "n_variables": n,
            "Q_shape": (n, n),
            "Q_entries": n * n,
            "Q_size_MB": round(n * n * 8 / 1e6, 4),   # float64
            "state_space_bits": n,
            "state_space_approx": f"2^{n} ≈ {2**n:,}" if n <= 40 else f"2^{n} (too large to enumerate)",
        }

    def print_info(self) -> None:
        """Print a formatted summary of QUBO problem dimensions."""
        d = self.info()
        print("FluxQ QUBO Problem Info")
        print("=" * 36)
        for k, v in d.items():
            print(f"  {k:<22} {v}")