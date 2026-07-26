"""
Tests for fluxq.qubo.builder and fluxq.qubo.objective.

Each test is a precise numerical check that pins down the exact QUBO
energy expected for a hand-computed scenario. These tests exist to catch
regressions in the mathematical encoding — if any test breaks, the QUBO
formulation has drifted from the spec.

Run with:
    python -m pytest tests/test_qubo_builder.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import Generator, make_toy_generators
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.objective import add_fuel_cost, add_startup_cost, add_shutdown_cost
from fluxq.qubo.validator import evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def single_gen(P_max=100.0, c_fuel=30.0, SU_cost=200.0, SD_cost=100.0) -> Generator:
    return Generator("test_gen", P_max=P_max, c_fuel=c_fuel, SU_cost=SU_cost, SD_cost=SD_cost)


# ---------------------------------------------------------------------------
# Objective — fuel cost
# ---------------------------------------------------------------------------

class TestFuelCost:
    def test_single_gen_single_timestep_on(self):
        """Single gen, T=1, ON: fuel = c_fuel * P_max * dt"""
        gen = single_gen()
        Q = np.zeros((1, 1))
        add_fuel_cost(Q, [gen], T=1, dt=1.0)
        expected = gen.c_fuel * gen.P_max * 1.0   # 3000.0
        assert abs(evaluate(Q, np.array([1.0])) - expected) < 1e-9

    def test_single_gen_single_timestep_off(self):
        """OFF state always contributes zero fuel cost."""
        gen = single_gen()
        Q = np.zeros((1, 1))
        add_fuel_cost(Q, [gen], T=1, dt=1.0)
        assert evaluate(Q, np.array([0.0])) == 0.0

    def test_fuel_cost_scales_with_dt(self):
        """Fuel cost is proportional to timestep duration."""
        gen = single_gen()
        for dt in [0.25, 0.5, 1.0, 2.0]:
            Q = np.zeros((1, 1))
            add_fuel_cost(Q, [gen], T=1, dt=dt)
            expected = gen.c_fuel * gen.P_max * dt
            assert abs(evaluate(Q, np.array([1.0])) - expected) < 1e-9

    def test_multi_gen_fuel_is_additive(self):
        """Fuel costs for independent generators sum independently."""
        gens = make_toy_generators()  # N=3
        T = 1
        Q = np.zeros((3, 1))   # n_vars = 3
        Q = np.zeros((3, 3))
        add_fuel_cost(Q, gens, T=T, dt=1.0)

        # Only gen0 ON
        x0 = np.array([1.0, 0.0, 0.0])
        expected0 = gens[0].c_fuel * gens[0].P_max
        assert abs(evaluate(Q, x0) - expected0) < 1e-9

        # Only gen1 ON
        x1 = np.array([0.0, 1.0, 0.0])
        expected1 = gens[1].c_fuel * gens[1].P_max
        assert abs(evaluate(Q, x1) - expected1) < 1e-9

        # Both gen0 and gen1 ON
        x01 = np.array([1.0, 1.0, 0.0])
        assert abs(evaluate(Q, x01) - (expected0 + expected1)) < 1e-9

    def test_fuel_cost_two_timesteps(self):
        """ON at both timesteps: double the fuel cost."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_fuel_cost(Q, [gen], T=T, dt=1.0)
        x_both = np.ones(T)
        expected = 2 * gen.c_fuel * gen.P_max
        assert abs(evaluate(Q, x_both) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Objective — startup cost
# ---------------------------------------------------------------------------

class TestStartupCost:
    def test_startup_at_t0_from_off(self):
        """Starting at t=0 from OFF initial state triggers SU_cost."""
        gen = single_gen()
        Q = np.zeros((1, 1))
        add_startup_cost(Q, [gen], T=1, initial_state=[0])
        assert abs(evaluate(Q, np.array([1.0])) - gen.SU_cost) < 1e-9
        assert evaluate(Q, np.array([0.0])) == 0.0

    def test_no_startup_at_t0_from_on(self):
        """No startup at t=0 if generator was already ON."""
        gen = single_gen()
        Q = np.zeros((1, 1))
        add_startup_cost(Q, [gen], T=1, initial_state=[1])
        # Was already ON, x=1 → no startup event
        assert evaluate(Q, np.array([1.0])) == 0.0
        assert evaluate(Q, np.array([0.0])) == 0.0

    def test_startup_at_t1(self):
        """Startup at t=1: OFF at t=0, ON at t=1."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_startup_cost(Q, [gen], T=T, initial_state=[0])
        x = np.array([0.0, 1.0])   # OFF at t=0, ON at t=1
        expected = gen.SU_cost
        assert abs(evaluate(Q, x) - expected) < 1e-9

    def test_no_startup_if_stays_on(self):
        """ON at t=0 and t=1: startup only at t=0 (initial=OFF)."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_startup_cost(Q, [gen], T=T, initial_state=[0])
        x = np.array([1.0, 1.0])
        expected = gen.SU_cost
        assert abs(evaluate(Q, x) - expected) < 1e-9

    def test_two_startups(self):
        """OFF→ON at t=0 and OFF→ON at t=2 (T=3): two startup events."""
        gen = single_gen()
        T = 3
        Q = np.zeros((T, T))
        add_startup_cost(Q, [gen], T=T, initial_state=[0])
        x = np.array([1.0, 0.0, 1.0])
        expected = 2 * gen.SU_cost
        assert abs(evaluate(Q, x) - expected) < 1e-9

    def test_startup_four_transitions(self):
        """Verify all 4 transitions for a 2-timestep problem."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_startup_cost(Q, [gen], T=T, initial_state=[0])
        SU = gen.SU_cost
        cases = [
            (np.array([0.0, 0.0]), 0.0,  "00: no startup"),
            (np.array([1.0, 0.0]), SU,   "10: startup at t0"),
            (np.array([0.0, 1.0]), SU,   "01: startup at t1"),
            (np.array([1.0, 1.0]), SU,   "11: startup only at t0"),
        ]
        for x, expected, desc in cases:
            result = evaluate(Q, x)
            assert abs(result - expected) < 1e-9, (
                f"{desc}: expected {expected}, got {result}"
            )


# ---------------------------------------------------------------------------
# Objective — shutdown cost
# ---------------------------------------------------------------------------

class TestShutdownCost:
    def test_shutdown_at_t1(self):
        """ON at t=0, OFF at t=1: shutdown cost."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_shutdown_cost(Q, [gen], T=T, initial_state=[0])
        x = np.array([1.0, 0.0])
        expected = gen.SD_cost
        assert abs(evaluate(Q, x) - expected) < 1e-9

    def test_no_shutdown_if_stays_off(self):
        """OFF throughout: no shutdown cost."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_shutdown_cost(Q, [gen], T=T, initial_state=[0])
        assert evaluate(Q, np.zeros(T)) == 0.0

    def test_no_shutdown_if_turns_on(self):
        """OFF→ON at t=1: no shutdown."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_shutdown_cost(Q, [gen], T=T, initial_state=[0])
        x = np.array([0.0, 1.0])
        assert evaluate(Q, x) == 0.0

    def test_no_shutdown_if_stays_on(self):
        """ON throughout: no shutdown."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_shutdown_cost(Q, [gen], T=T, initial_state=[0])
        x = np.ones(T)
        assert abs(evaluate(Q, x)) < 1e-9

    def test_shutdown_four_transitions(self):
        """Verify all 4 transitions for a 2-timestep problem."""
        gen = single_gen()
        T = 2
        Q = np.zeros((T, T))
        add_shutdown_cost(Q, [gen], T=T, initial_state=[0])
        SD = gen.SD_cost
        cases = [
            (np.array([0.0, 0.0]), 0.0,  "00: no shutdown"),
            (np.array([1.0, 0.0]), SD,   "10: shutdown at t1"),
            (np.array([0.0, 1.0]), 0.0,  "01: no shutdown (turned on)"),
            (np.array([1.0, 1.0]), 0.0,  "11: no shutdown"),
        ]
        for x, expected, desc in cases:
            result = evaluate(Q, x)
            assert abs(result - expected) < 1e-9, (
                f"{desc}: expected {expected}, got {result}"
            )


# ---------------------------------------------------------------------------
# Combined objective (fuel + startup + shutdown)
# ---------------------------------------------------------------------------

class TestCombinedObjective:
    def test_combined_all_four_transitions(self):
        """
        Full 2-timestep problem with all three cost components.
        Exhaustive check of all 4 binary states.
        """
        gen = Generator("g", P_max=100.0, c_fuel=30.0, SU_cost=200.0, SD_cost=100.0)
        T   = 2
        dt  = 1.0
        builder = QUBOBuilder([gen], T=T, dt=dt)
        # No power balance (lambda_1=0)
        demand = np.array([0.0, 0.0])
        Q = builder.build(demand, LambdaConfig(lambda_1=0.0))

        F  = gen.c_fuel * gen.P_max * dt   # fuel per timestep = 3000
        SU = gen.SU_cost                    # 200
        SD = gen.SD_cost                    # 100

        # State 00: no fuel, no startup, no shutdown
        assert abs(builder.evaluate(np.array([0.0, 0.0])) - 0.0) < 1e-9
        # State 10: ON at t=0, OFF at t=1
        assert abs(builder.evaluate(np.array([1.0, 0.0])) - (F + SU + SD)) < 1e-9
        # State 01: OFF at t=0, ON at t=1
        assert abs(builder.evaluate(np.array([0.0, 1.0])) - (F + SU)) < 1e-9
        # State 11: ON at both
        assert abs(builder.evaluate(np.array([1.0, 1.0])) - (2 * F + SU)) < 1e-9

    def test_cheaper_to_stay_on_than_cycle(self):
        """
        ON-OFF-ON costs more than ON-ON-ON (startup + shutdown + startup vs just running).
        Verifies that SU+SD costs are large enough to discourage cycling.
        """
        gen = Generator("g", P_max=100.0, c_fuel=30.0, SU_cost=500.0, SD_cost=300.0)
        T = 3
        builder = QUBOBuilder([gen], T=T)
        demand = np.zeros(T)
        Q = builder.build(demand, LambdaConfig(lambda_1=0.0))

        e_cycle    = builder.evaluate(np.array([1.0, 0.0, 1.0]))  # ON-OFF-ON
        e_stay_on  = builder.evaluate(np.array([1.0, 1.0, 1.0]))  # ON-ON-ON

        assert e_cycle < e_stay_on, (
            f"Expected cycling ({e_cycle:.0f}) < staying on ({e_stay_on:.0f})"
        )


# ---------------------------------------------------------------------------
# QUBOBuilder — structure and metadata
# ---------------------------------------------------------------------------

class TestQUBOBuilder:
    def test_shape(self):
        """Q has shape (N*T, N*T)."""
        gens = make_toy_generators()   # N=3
        T = 4
        builder = QUBOBuilder(gens, T=T)
        Q = builder.build(np.zeros(T), LambdaConfig(lambda_1=0.0))
        assert Q.shape == (12, 12)

    def test_upper_triangular(self):
        """Q must be strictly upper triangular (no entries below diagonal)."""
        gens = make_toy_generators()
        T = 4
        builder = QUBOBuilder(gens, T=T)
        Q = builder.build(np.array([200.0, 250.0, 300.0, 200.0]))
        lower_norm = np.linalg.norm(np.tril(Q, k=-1))
        assert lower_norm < 1e-12, f"Lower triangle norm = {lower_norm:.2e}"

    def test_var_idx(self):
        """var_idx maps (i, t) correctly."""
        builder = QUBOBuilder(make_toy_generators(), T=4)
        assert builder.var_idx(0, 0) == 0
        assert builder.var_idx(0, 3) == 3
        assert builder.var_idx(1, 0) == 4
        assert builder.var_idx(2, 3) == 11

    def test_idx_to_gen_time_roundtrip(self):
        """idx_to_gen_time is the inverse of var_idx."""
        builder = QUBOBuilder(make_toy_generators(), T=4)
        for i in range(3):
            for t in range(4):
                idx = builder.var_idx(i, t)
                i2, t2 = builder.idx_to_gen_time(idx)
                assert (i2, t2) == (i, t)

    def test_suggest_lambda_positive(self):
        """suggest_lambda() always returns a positive float."""
        builder = QUBOBuilder(make_toy_generators(), T=4)
        lam = builder.suggest_lambda()
        assert lam > 0.0

    def test_all_off_zero_objective(self):
        """All-OFF has zero QUBO energy when lambda_1=0."""
        gens = make_toy_generators()
        T = 4
        builder = QUBOBuilder(gens, T=T)
        Q = builder.build(np.zeros(T), LambdaConfig(lambda_1=0.0))
        e = builder.evaluate(np.zeros(builder.n_vars))
        assert abs(e) < 1e-12

    def test_build_raises_on_wrong_demand_shape(self):
        """build() raises ValueError if demand has wrong length."""
        builder = QUBOBuilder(make_toy_generators(), T=4)
        with pytest.raises(ValueError):
            builder.build(np.array([1.0, 2.0]))   # wrong length

    def test_constructor_raises_on_empty_generators(self):
        """QUBOBuilder raises on empty generator list."""
        with pytest.raises(ValueError):
            QUBOBuilder([], T=4)

    def test_constructor_raises_on_T_zero(self):
        with pytest.raises(ValueError):
            QUBOBuilder(make_toy_generators(), T=0)
