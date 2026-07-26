"""
Tests for fluxq.qubo.constraints.

Each test verifies the QUBO energy of a specific binary state matches the
analytically-computed penalty value, confirming the constraint encoding is
correct.

Run with:
    python -m pytest tests/test_constraints.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import Generator, make_toy_generators
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.constraints import add_power_balance, add_spinning_reserve
from fluxq.qubo.validator import evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def two_gens():
    """Two generators: 100 MW and 200 MW."""
    return [
        Generator("G1", P_max=100.0, c_fuel=30.0, SU_cost=0.0, SD_cost=0.0),
        Generator("G2", P_max=200.0, c_fuel=20.0, SU_cost=0.0, SD_cost=0.0),
    ]


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------

class TestPowerBalance:
    """
    Power balance penalty: λ · (Σ P_i x_i − D)²

    We disable startup/shutdown costs (SU=SD=0) and set lambda_1 explicitly
    so the penalty value is easy to compute by hand.
    """

    def test_exact_balance_is_minimum_energy(self):
        """
        Single generator at full power = demand → minimum QUBO energy.

        Note: QUBO drops the constant D² term, so E(x=1) ≠ 0 even when the
        generator exactly matches demand. What matters is that x=1 (balanced)
        has strictly lower energy than x=0 (deficit = demand).
        """
        gen = Generator("G", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0)
        T = 1
        demand = np.array([100.0])
        lam = 1.0
        Q = np.zeros((1, 1))
        add_power_balance(Q, [gen], demand, T, lam)
        e_on  = evaluate(Q, np.array([1.0]))
        e_off = evaluate(Q, np.array([0.0]))
        assert e_on < e_off, (
            f"Balanced (E={e_on:.1f}) should be lower than deficit (E={e_off:.1f})"
        )
        expected_diag = lam * 100.0 * (100.0 - 2 * 100.0)   # = -10000
        assert abs(Q[0, 0] - expected_diag) < 1e-9

    def test_shortage_penalty(self):
        """Generator off, demand = 100 → penalty = λ·(0−100)² = λ·10000 (constant dropped)."""
        gen = Generator("G", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0)
        T = 1
        demand = np.array([100.0])
        lam = 1.0
        Q = np.zeros((1, 1))
        add_power_balance(Q, [gen], demand, T, lam)
        e_off = evaluate(Q, np.array([0.0]))
        e_on  = evaluate(Q, np.array([1.0]))
        assert abs(Q[0, 0] - (-10000.0)) < 1e-9
        assert abs(e_off - 0.0) < 1e-9
        assert abs(e_on - (-10000.0)) < 1e-9

    def test_two_gen_relative_ordering(self):
        """
        With 2 generators (100 MW, 200 MW) and demand = 150 MW:
         - Both off:  penalty = λ · 150² = 22500λ  [but constant dropped → 0]
         - G1 only:   penalty = λ · (100-150)² = 2500λ → QUBO = 2500λ - 22500λ = -20000λ
         - G2 only:   penalty = λ · (200-150)² = 2500λ → QUBO = -20000λ
         - Both on:   penalty = λ · (300-150)² = 22500λ → QUBO = 0

        G1 and G2 alone tie at best; both ON is worst.
        """
        gens = two_gens()
        T = 1
        demand = np.array([150.0])
        lam = 1.0
        n = 2
        Q = np.zeros((n, n))
        add_power_balance(Q, gens, demand, T, lam)

        e_off  = evaluate(Q, np.array([0.0, 0.0]))
        e_g1   = evaluate(Q, np.array([1.0, 0.0]))
        e_g2   = evaluate(Q, np.array([0.0, 1.0]))
        e_both = evaluate(Q, np.array([1.0, 1.0]))

        assert abs(e_off  - 0.0)     < 1e-9
        assert abs(e_g1   - (-20000.0)) < 1e-9
        assert abs(e_g2   - (-20000.0)) < 1e-9
        assert abs(e_both - 0.0)     < 1e-9

        assert abs(e_g1 - e_g2) < 1e-9
        assert abs(e_both - e_off) < 1e-9

    def test_two_gen_one_better(self):
        """
        G1=100 MW, G2=200 MW, demand=200 MW:
         G2 exactly meets demand → zero penalty (relative).
         G1 has 100 MW shortage.
        """
        gens = two_gens()
        T = 1
        demand = np.array([200.0])
        lam = 1.0
        Q = np.zeros((2, 2))
        add_power_balance(Q, gens, demand, T, lam)

        e_g1   = evaluate(Q, np.array([1.0, 0.0]))   # 100 MW, shortage 100
        e_g2   = evaluate(Q, np.array([0.0, 1.0]))   # 200 MW, exact match
        e_both = evaluate(Q, np.array([1.0, 1.0]))   # 300 MW, excess 100

        assert e_g2 < e_g1
        assert e_g2 < e_both
        assert abs(e_g1 - e_both) < 1e-9

    def test_multi_timestep_independent(self):
        """Power balance penalty is summed independently per timestep."""
        gen = Generator("G", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0)
        T = 3
        demand = np.array([100.0, 100.0, 100.0])   # all matched when ON
        lam = 1.0
        Q = np.zeros((T, T))
        add_power_balance(Q, [gen], demand, T, lam)
        e_all_on  = evaluate(Q, np.ones(T))
        e_all_off = evaluate(Q, np.zeros(T))
        assert e_all_on < e_all_off

    def test_lambda_scales_penalty(self):
        """Doubling lambda doubles the penalty magnitude."""
        gen = Generator("G", P_max=100.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0)
        demand = np.array([50.0])   # 50 MW demand, generator gives 100 MW → imbal = 50

        Q1 = np.zeros((1, 1))
        Q2 = np.zeros((1, 1))
        add_power_balance(Q1, [gen], demand, 1, lambda_1=1.0)
        add_power_balance(Q2, [gen], demand, 1, lambda_1=2.0)
        x = np.array([1.0])
        assert abs(Q2[0, 0] - 2 * Q1[0, 0]) < 1e-9

    def test_via_builder(self):
        """End-to-end: power balance via QUBOBuilder."""
        gens = make_toy_generators()   # N=3, capacities: 200, 150, 100 MW
        T = 1
        demand = np.array([350.0])     # exactly coal+gas = 350 MW
        lam = 1.0
        builder = QUBOBuilder(gens, T=T)
        # Disable objective costs so only power balance matters
        builder.generators = [
            Generator(g.name, g.P_max, 0.0, 0.0, 0.0)
            for g in gens
        ]
        Q = builder.build(demand, LambdaConfig(lambda_1=lam))

        x_coal_gas = np.array([1.0, 1.0, 0.0])
        e_coal_gas  = builder.evaluate(x_coal_gas)

        x_coal = np.array([1.0, 0.0, 0.0])
        e_coal  = builder.evaluate(x_coal)

        x_all  = np.ones(3)
        e_all   = builder.evaluate(x_all)

        assert e_coal_gas < e_coal
        assert e_coal_gas < e_all


# ---------------------------------------------------------------------------
# Spinning reserve
# ---------------------------------------------------------------------------

class TestSpinningReserve:
    def test_reserve_adds_to_target(self):
        """
        Reserve penalty uses demand + reserve as target.

        QUBO energies can't be compared directly between different targets
        because the dropped constant D² differs. We add the constant back to
        get actual penalty values and compare those.
        """
        gen = Generator("G", P_max=300.0, c_fuel=0.0, SU_cost=0.0, SD_cost=0.0)
        T = 1
        demand  = np.array([200.0])
        reserve = np.array([50.0])
        lam = 1.0

        Q_res = np.zeros((1, 1))
        add_spinning_reserve(Q_res, [gen], demand, reserve, T, lambda_2=lam)

        Q_bal = np.zeros((1, 1))
        add_power_balance(Q_bal, [gen], demand, T, lambda_1=lam)

        x = np.array([1.0])

        target_res = demand[0] + reserve[0]   # = 250
        target_bal = demand[0]                # = 200

        actual_penalty_res = evaluate(Q_res, x) + lam * target_res ** 2  # (300-250)² = 2500
        actual_penalty_bal = evaluate(Q_bal, x) + lam * target_bal ** 2  # (300-200)² = 10000

        assert abs(actual_penalty_res - lam * 50.0 ** 2) < 1e-9,  \
            f"Reserve penalty should be λ·50²=2500, got {actual_penalty_res}"
        assert abs(actual_penalty_bal - lam * 100.0 ** 2) < 1e-9, \
            f"Balance penalty should be λ·100²=10000, got {actual_penalty_bal}"

        assert actual_penalty_res < actual_penalty_bal


# ---------------------------------------------------------------------------
# Phase 1 stubs raise NotImplementedError
# ---------------------------------------------------------------------------

class TestPhase1Stubs:
    def test_mut_raises(self):
        from fluxq.qubo.constraints import add_mut_constraint
        Q = np.zeros((4, 4))
        with pytest.raises(NotImplementedError):
            add_mut_constraint(Q, make_toy_generators(), T=4, lambda_3=1.0)

    def test_mdt_raises(self):
        from fluxq.qubo.constraints import add_mdt_constraint
        Q = np.zeros((4, 4))
        with pytest.raises(NotImplementedError):
            add_mdt_constraint(Q, make_toy_generators(), T=4, lambda_4=1.0)

    def test_ramp_raises(self):
        from fluxq.qubo.constraints import add_ramp_constraint
        Q = np.zeros((4, 4))
        with pytest.raises(NotImplementedError):
            add_ramp_constraint(Q, make_toy_generators(), T=4, lambda_5=1.0)
