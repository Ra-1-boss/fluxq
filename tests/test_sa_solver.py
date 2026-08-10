"""
Tests for fluxq.solvers.sa_solver.

Simulated Annealing is FluxQ's first heuristic solver -- unlike every
QUBO-formulation test so far, there is no hand-derived formula to check
it against. Instead, every claim of correctness here is anchored to
fluxq.solvers.brute_force, which gives a true, exhaustively-proven
optimum on problems small enough to enumerate (n_vars <= 25).

Iteration counts used below were chosen empirically, not guessed: an
investigation logged in docs/research-log.md found that the toy problem
(n_vars=12, energy scale in the thousands) converges to the exact
optimum reliably with 5,000 iterations, while the real RTS-GMLC problem
(n_vars=20, energy scale in the billions, from real dollar costs) needed
20,000 to close a small but real gap that 8,000 left behind. Both
numbers are used here rather than picking one arbitrarily for both.

Run with:
    python -m pytest tests/test_sa_solver.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import make_toy_generators, make_rts_gmlc_sample
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.solvers.brute_force import brute_force
from fluxq.solvers.sa_solver import simulated_annealing, estimate_initial_temperature


# ---------------------------------------------------------------------------
# Correctness against brute-force ground truth
# ---------------------------------------------------------------------------

class TestSAFindsKnownOptimum:
    def test_toy_problem_exact_match(self):
        """
        The original Phase 0 toy problem (3 generators, T=4). Brute force
        gives the true optimum; SA with 5 restarts / 5,000 iterations
        each should find the exact same energy AND the exact same
        schedule -- not just a close answer.
        """
        generators = make_toy_generators()
        T = 4
        demand = np.array([200.0, 280.0, 350.0, 220.0])
        builder = QUBOBuilder(generators, T=T)
        Q = builder.build(demand, LambdaConfig(lambda_1=5.0))

        true_energy, true_x = brute_force(Q, builder.n_vars, top_k=1)[0]

        sa_result = simulated_annealing(
            Q, builder.n_vars, n_iterations=5000, n_restarts=5, seed=42
        )

        assert abs(sa_result["best_energy"] - true_energy) < 1e-6, (
            f"SA energy {sa_result['best_energy']:.4f} != "
            f"true optimum {true_energy:.4f}"
        )
        assert np.array_equal(sa_result["best_x"], true_x), (
            "SA found the right energy but a different schedule -- "
            "possible degenerate optimum, worth investigating."
        )

    def test_real_rts_gmlc_5gen_exact_match_with_sufficient_iterations(self):
        """
        Real RTS-GMLC 5-generator sample, same demand profile used
        throughout the research log. Requires 20,000 iterations (not
        5,000 or 8,000 -- both were tried and left a small but real gap;
        see docs/research-log.md) because this problem's energy scale is
        vastly larger (real dollar costs, billions) than the toy problem's.
        """
        generators = make_rts_gmlc_sample()
        T = 4
        demand = np.array([350.0, 705.0, 856.0, 426.0])
        builder = QUBOBuilder(generators, T=T)
        lam = builder.suggest_lambda()
        Q = builder.build(demand, LambdaConfig(lambda_1=lam * 1.5))

        true_energy, true_x = brute_force(Q, builder.n_vars, top_k=1)[0]

        sa_result = simulated_annealing(
            Q, builder.n_vars, n_iterations=20000, n_restarts=5, seed=7
        )

        assert abs(sa_result["best_energy"] - true_energy) < 1e-3, (
            f"SA energy {sa_result['best_energy']:.4f} != "
            f"true optimum {true_energy:.4f}"
        )

    def test_insufficient_iterations_honestly_falls_short(self):
        """
        This test exists specifically to document, not hide, the finding
        that fewer iterations leaves a real (if small) gap on this
        problem. If this test starts failing because 8,000 iterations
        suddenly closes the gap, that's a meaningful change in the
        algorithm worth investigating -- not something to just delete.
        """
        generators = make_rts_gmlc_sample()
        T = 4
        demand = np.array([350.0, 705.0, 856.0, 426.0])
        builder = QUBOBuilder(generators, T=T)
        lam = builder.suggest_lambda()
        Q = builder.build(demand, LambdaConfig(lambda_1=lam * 1.5))

        true_energy, _ = brute_force(Q, builder.n_vars, top_k=1)[0]

        sa_result = simulated_annealing(
            Q, builder.n_vars, n_iterations=8000, n_restarts=5, seed=7
        )
        gap = sa_result["best_energy"] - true_energy
        relative_gap = abs(gap) / abs(true_energy)

        # The gap should be small relative to the problem's billion-scale
        # energy, but real and nonzero at this iteration count. Bound set
        # from the actual value observed during investigation (~18,555),
        # with margin -- not a placeholder guess.
        assert 0 < abs(gap) < 30000, (
            f"Expected a nonzero gap under 30,000 at 8,000 iterations, got {gap}"
        )
        assert relative_gap < 1e-4


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_identical_result(self):
        generators = make_toy_generators()
        builder = QUBOBuilder(generators, T=4)
        Q = builder.build(np.array([200.0, 280.0, 350.0, 220.0]), LambdaConfig(lambda_1=5.0))

        r1 = simulated_annealing(Q, builder.n_vars, n_iterations=2000, n_restarts=3, seed=99)
        r2 = simulated_annealing(Q, builder.n_vars, n_iterations=2000, n_restarts=3, seed=99)

        assert r1["best_energy"] == r2["best_energy"]
        assert np.array_equal(r1["best_x"], r2["best_x"])

    def test_different_seeds_can_give_different_restart_spread(self):
        """
        Not asserting they MUST differ (they could coincidentally agree)
        -- just confirming the seed is actually being used, not ignored.
        """
        generators = make_toy_generators()
        builder = QUBOBuilder(generators, T=4)
        Q = builder.build(np.array([200.0, 280.0, 350.0, 220.0]), LambdaConfig(lambda_1=5.0))

        r1 = simulated_annealing(Q, builder.n_vars, n_iterations=500, n_restarts=1, seed=1)
        r2 = simulated_annealing(Q, builder.n_vars, n_iterations=500, n_restarts=1, seed=2)
        # With very few iterations and a single restart each, these should
        # very likely differ -- not a hard guarantee, but a reasonable check.
        assert r1["all_energies"] != r2["all_energies"] or not np.array_equal(
            r1["best_x"], r2["best_x"]
        )


# ---------------------------------------------------------------------------
# estimate_initial_temperature
# ---------------------------------------------------------------------------

class TestEstimateInitialTemperature:
    def test_returns_positive_float(self):
        generators = make_toy_generators()
        builder = QUBOBuilder(generators, T=4)
        Q = builder.build(np.array([200.0, 280.0, 350.0, 220.0]), LambdaConfig(lambda_1=5.0))
        T0 = estimate_initial_temperature(Q, builder.n_vars, seed=1)
        assert T0 > 0

    def test_deterministic_with_seed(self):
        generators = make_toy_generators()
        builder = QUBOBuilder(generators, T=4)
        Q = builder.build(np.array([200.0, 280.0, 350.0, 220.0]), LambdaConfig(lambda_1=5.0))
        T0_a = estimate_initial_temperature(Q, builder.n_vars, seed=5)
        T0_b = estimate_initial_temperature(Q, builder.n_vars, seed=5)
        assert T0_a == T0_b


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_raises_on_zero_n_vars(self):
        Q = np.zeros((0, 0))
        with pytest.raises(ValueError):
            simulated_annealing(Q, 0)

    def test_raises_on_zero_iterations(self):
        Q = np.zeros((3, 3))
        with pytest.raises(ValueError):
            simulated_annealing(Q, 3, n_iterations=0)

    def test_raises_on_zero_restarts(self):
        Q = np.zeros((3, 3))
        with pytest.raises(ValueError):
            simulated_annealing(Q, 3, n_restarts=0)

    def test_raises_when_T0_not_greater_than_T_min(self):
        Q = np.zeros((3, 3))
        with pytest.raises(ValueError):
            simulated_annealing(Q, 3, T0=0.001, T_min=0.01)
