"""
Tests for fluxq.qubo.constraints.add_mut_constraint, add_mdt_constraint,
and add_transition_linking.

Every expected value in this file was worked out by hand first -- see
docs/research-log.md for the full derivation, including a matrix-ordering
bug caught during that process (an early draft wrote quadratic terms
involving the auxiliary w variable in the wrong row/column order, which
would have silently broken the "Q is strictly upper-triangular" structural
guarantee even though the computed energy values would still have been
numerically correct).

Run with:
    python -m pytest tests/test_mut_mdt_constraints.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import Generator, make_rts_gmlc_sample
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.constraints import add_transition_linking, add_mut_constraint, add_mdt_constraint


# ---------------------------------------------------------------------------
# Hand-derived expected tables
# ---------------------------------------------------------------------------
# Both tables below were worked out by hand, independently of any code,
# before this file was written. See docs/research-log.md for the full
# real-world feasibility reasoning behind each entry.

# MUT=2, single generator, T=3, starts OFF (initial_state=0).
# Key: (x0, x1, x2) -> expected penalty (with w set to the TRUE product).
MUT2_EXPECTED = {
    (0, 0, 0): 0,   # never starts -- feasible
    (1, 0, 0): 1,   # starts at t0, off after 1 step -- INFEASIBLE
    (0, 1, 0): 1,   # starts at t1, off after 1 step -- INFEASIBLE
    (0, 0, 1): 0,   # starts too late to check within horizon -- feasible
    (1, 1, 0): 0,   # starts at t0, stays 2 steps -- feasible
    (1, 0, 1): 1,   # starts at t0, off after 1 step -- INFEASIBLE
    (0, 1, 1): 0,   # starts at t1, stays 2 steps -- feasible
    (1, 1, 1): 0,   # feasible
}

# MDT=2, single generator, T=3, starts ON (initial_state=1). Symmetric
# derivation to MUT2_EXPECTED, worked out independently by hand.
MDT2_EXPECTED = {
    (0, 0, 0): 0,   # shuts at t0, stays off 2 steps -- feasible
    (1, 0, 0): 0,   # shuts at t1, stays off 2 steps -- feasible
    (0, 1, 0): 1,   # shuts at t0, restarts after 1 step -- INFEASIBLE
    (0, 0, 1): 0,   # shuts at t0, stays off long enough within horizon
    (1, 1, 0): 0,   # never shuts within a checkable window -- feasible
    (1, 0, 1): 1,   # shuts at t1, restarts after 1 step -- INFEASIBLE
    (0, 1, 1): 1,   # shuts at t0, restarts after 1 step -- INFEASIBLE
    (1, 1, 1): 0,   # feasible
}


def _energy_with_true_w(builder: QUBOBuilder, x0: int, x1: int, x2: int) -> float:
    """Evaluate QUBO energy for a 3-timestep, 1-generator schedule, with
    w set to the true product x_{t-1}*x_t at each transition."""
    w1 = x0 * x1
    w2 = x1 * x2
    x = np.array([x0, x1, x2, w1, w2], dtype=float)
    return builder.evaluate(x)


class TestMUTConstraint:
    def test_matches_hand_derived_table(self):
        """
        Every one of the 8 possible 3-timestep schedules must produce
        exactly the hand-derived penalty, with w set to the correct
        product -- confirms the formula itself is right.
        """
        gen = Generator("g", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0, MUT=2, MDT=1)
        builder = QUBOBuilder([gen], T=3, enable_mut_mdt=True)
        Q = builder.build(
            np.zeros(3),
            LambdaConfig(lambda_1=0.0, lambda_2=0.0, lambda_3=1.0, lambda_4=0.0, lambda_link=10.0),
        )
        for (x0, x1, x2), expected in MUT2_EXPECTED.items():
            e = _energy_with_true_w(builder, x0, x1, x2)
            assert abs(e - expected) < 1e-9, (
                f"Schedule {(x0, x1, x2)}: expected {expected}, got {e}"
            )

    def test_optimizer_is_incentivized_to_pick_correct_w(self):
        """
        The strongest form of verification: brute-force ALL 32 states
        (8 x-patterns x 4 w-combinations) and confirm the MINIMUM-energy
        w for every x-pattern is exactly the true product. This proves
        the linking penalty genuinely makes correct w optimal -- not
        just "correct when handed the right answer."
        """
        gen = Generator("g", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0, MUT=2, MDT=1)
        builder = QUBOBuilder([gen], T=3, enable_mut_mdt=True)
        Q = builder.build(
            np.zeros(3),
            LambdaConfig(lambda_1=0.0, lambda_2=0.0, lambda_3=1.0, lambda_4=0.0, lambda_link=10.0),
        )
        for x0 in (0, 1):
            for x1 in (0, 1):
                for x2 in (0, 1):
                    best_e, best_w = None, None
                    for w1 in (0, 1):
                        for w2 in (0, 1):
                            x = np.array([x0, x1, x2, w1, w2], dtype=float)
                            e = builder.evaluate(x)
                            if best_e is None or e < best_e:
                                best_e, best_w = e, (w1, w2)
                    true_w = (x0 * x1, x1 * x2)
                    assert best_w == true_w, (
                        f"Pattern {(x0,x1,x2)}: minimum-energy w was {best_w}, "
                        f"should be the true product {true_w}"
                    )
                    expected = MUT2_EXPECTED[(x0, x1, x2)]
                    assert abs(best_e - expected) < 1e-9

    def test_zero_or_one_mut_disables_constraint(self):
        """MUT<=1 should add no penalty terms at all -- always zero energy."""
        gen = Generator("g", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0, MUT=1, MDT=1)
        builder = QUBOBuilder([gen], T=3, enable_mut_mdt=True)
        Q = builder.build(
            np.zeros(3),
            LambdaConfig(lambda_1=0.0, lambda_2=0.0, lambda_3=5.0, lambda_4=0.0, lambda_link=10.0),
        )
        # Any cycling pattern should incur zero MUT penalty when MUT=1.
        e = _energy_with_true_w(builder, 1, 0, 1)
        assert abs(e) < 1e-9


class TestMDTConstraint:
    def test_matches_hand_derived_table(self):
        gen = Generator("g", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0, MUT=1, MDT=2)
        builder = QUBOBuilder([gen], T=3, initial_state=[1], enable_mut_mdt=True)
        Q = builder.build(
            np.zeros(3),
            LambdaConfig(lambda_1=0.0, lambda_2=0.0, lambda_3=0.0, lambda_4=1.0, lambda_link=10.0),
        )
        for (x0, x1, x2), expected in MDT2_EXPECTED.items():
            e = _energy_with_true_w(builder, x0, x1, x2)
            assert abs(e - expected) < 1e-9, (
                f"Schedule {(x0, x1, x2)}: expected {expected}, got {e}"
            )

    def test_optimizer_is_incentivized_to_pick_correct_w(self):
        gen = Generator("g", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0, MUT=1, MDT=2)
        builder = QUBOBuilder([gen], T=3, initial_state=[1], enable_mut_mdt=True)
        Q = builder.build(
            np.zeros(3),
            LambdaConfig(lambda_1=0.0, lambda_2=0.0, lambda_3=0.0, lambda_4=1.0, lambda_link=10.0),
        )
        for x0 in (0, 1):
            for x1 in (0, 1):
                for x2 in (0, 1):
                    best_e, best_w = None, None
                    for w1 in (0, 1):
                        for w2 in (0, 1):
                            x = np.array([x0, x1, x2, w1, w2], dtype=float)
                            e = builder.evaluate(x)
                            if best_e is None or e < best_e:
                                best_e, best_w = e, (w1, w2)
                    true_w = (x0 * x1, x1 * x2)
                    assert best_w == true_w
                    expected = MDT2_EXPECTED[(x0, x1, x2)]
                    assert abs(best_e - expected) < 1e-9


class TestStructuralInvariants:
    def test_q_remains_upper_triangular_with_mut_mdt(self):
        """
        This is the exact invariant a matrix-ordering bug would break --
        it was caught during derivation before this test even existed.
        """
        generators = make_rts_gmlc_sample()
        builder = QUBOBuilder(generators, T=4, enable_mut_mdt=True)
        demand = np.array([350.0, 705.0, 856.0, 426.0])
        lam = builder.suggest_lambda()
        lam_link = builder.suggest_lambda_link(lambda_3=1.0, lambda_4=1.0)
        Q = builder.build(
            demand,
            LambdaConfig(lambda_1=lam * 1.5, lambda_3=1.0, lambda_4=1.0, lambda_link=lam_link),
        )
        lower_norm = np.linalg.norm(np.tril(Q, k=-1))
        assert lower_norm < 1e-9, f"Q is not upper-triangular (lower norm = {lower_norm:.2e})"

    def test_n_vars_grows_correctly_for_real_generators(self):
        """
        n_vars should equal N*T + N*(T-1) exactly when MUT/MDT is enabled --
        checked here against the real 5-generator RTS-GMLC sample rather
        than a synthetic one.
        """
        generators = make_rts_gmlc_sample()
        N, T = len(generators), 4
        builder = QUBOBuilder(generators, T=T, enable_mut_mdt=True)
        assert builder.n_vars_real == N * T
        assert builder.n_vars_w == N * (T - 1)
        assert builder.n_vars == N * T + N * (T - 1)

    def test_disabled_by_default_matches_phase_0_exactly(self):
        """
        enable_mut_mdt defaults to False -- n_vars, and every Phase 0
        test that already passed, must be completely unaffected.
        """
        generators = make_rts_gmlc_sample()
        builder = QUBOBuilder(generators, T=4)
        assert builder.enable_mut_mdt is False
        assert builder.n_vars == builder.n_vars_real
        assert builder.n_vars_w == 0

    def test_suggest_lambda_link_raises_when_disabled(self):
        generators = make_rts_gmlc_sample()
        builder = QUBOBuilder(generators, T=4, enable_mut_mdt=False)
        with pytest.raises(RuntimeError):
            builder.suggest_lambda_link()

    def test_w_idx_raises_when_disabled(self):
        generators = make_rts_gmlc_sample()
        builder = QUBOBuilder(generators, T=4, enable_mut_mdt=False)
        with pytest.raises(RuntimeError):
            builder.w_idx(0, 1)
