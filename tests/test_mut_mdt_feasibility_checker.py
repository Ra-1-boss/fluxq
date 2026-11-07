"""
Tests for fluxq.qubo.validator.check_mut_mdt_feasibility.

This is FluxQ's second, independent method for checking MUT/MDT
compliance -- it never touches the QUBO matrix, only the decoded 0/1
schedule. Every expected value here matches the same hand-derived tables
used in test_mut_mdt_constraints.py, confirming the two independent
methods (QUBO-penalty-based and schedule-walk-based) agree on every case.

The two edge cases at the bottom exist specifically because an early
draft of the checker had a real bug -- conflating "the run reached the
end of the horizon" with "we can't tell if this run was long enough" --
which happened to produce the right final answer on the 16 basic cases
by coincidence, for the wrong reason. These two cases are the ones that
actually distinguish correct reasoning from the coincidence. See
docs/research-log.md for the full story.

Run with:
    python -m pytest tests/test_mut_mdt_feasibility_checker.py -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import Generator
from fluxq.qubo.validator import check_mut_mdt_feasibility


# Same tables as test_mut_mdt_constraints.py -- MUT=2, T=3, starts OFF.
MUT2_EXPECTED_FEASIBLE = {
    (0, 0, 0): True,  (1, 0, 0): False, (0, 1, 0): False, (0, 0, 1): True,
    (1, 1, 0): True,  (1, 0, 1): False, (0, 1, 1): True,  (1, 1, 1): True,
}

# MDT=2, T=3, starts ON.
MDT2_EXPECTED_FEASIBLE = {
    (0, 0, 0): True,  (1, 0, 0): True,  (0, 1, 0): False, (0, 0, 1): True,
    (1, 1, 0): True,  (1, 0, 1): False, (0, 1, 1): False, (1, 1, 1): True,
}


class TestMUTFeasibility:
    def test_matches_hand_derived_table(self):
        gen = [Generator("g", 100.0, 0.0, 0.0, 0.0, MUT=2, MDT=1)]
        for (x0, x1, x2), expected in MUT2_EXPECTED_FEASIBLE.items():
            x = np.array([x0, x1, x2], dtype=float)
            result = check_mut_mdt_feasibility(x, gen, T=3, initial_state=[0])
            assert result["feasible"] == expected, (
                f"Pattern {(x0,x1,x2)}: expected feasible={expected}, "
                f"got {result['feasible']} -- {result['violations']}"
            )


class TestMDTFeasibility:
    def test_matches_hand_derived_table(self):
        gen = [Generator("g", 100.0, 0.0, 0.0, 0.0, MUT=1, MDT=2)]
        for (x0, x1, x2), expected in MDT2_EXPECTED_FEASIBLE.items():
            x = np.array([x0, x1, x2], dtype=float)
            result = check_mut_mdt_feasibility(x, gen, T=3, initial_state=[1])
            assert result["feasible"] == expected, (
                f"Pattern {(x0,x1,x2)}: expected feasible={expected}, "
                f"got {result['feasible']} -- {result['violations']}"
            )


class TestBoundaryReasoningIsCorrect:
    """
    These two cases specifically distinguish the fixed logic from the
    bug caught during derivation -- see module docstring.
    """

    def test_genuine_violation_not_a_boundary_case(self):
        """
        MUT=3, T=4, x=[1,1,0,0]: starts at t0, stays on for only 2 steps,
        THEN genuinely shuts off at t2 (fully observed, not a horizon
        cutoff). This must be flagged -- if the boundary logic were
        wrong in the other direction, a bug could incorrectly excuse it.
        """
        gen = [Generator("g", 100.0, 0.0, 0.0, 0.0, MUT=3, MDT=1)]
        x = np.array([1, 1, 0, 0], dtype=float)
        result = check_mut_mdt_feasibility(x, gen, T=4, initial_state=[0])
        assert result["feasible"] is False
        assert len(result["violations"]) == 1
        assert result["violations"][0]["type"] == "MUT"

    def test_run_confirmed_satisfied_even_though_it_reaches_horizon_end(self):
        """
        MUT=2, T=3, x=[0,1,1]: starts at t1, run reaches exactly the end
        of the horizon (t1, t2) with run_len=2=MUT. This must be
        classified as CONFIRMED SATISFIED (run_len >= min_required
        checked first), not accidentally-correct via "we couldn't check
        it" reasoning -- that's the exact bug that was caught and fixed.
        """
        gen = [Generator("g", 100.0, 0.0, 0.0, 0.0, MUT=2, MDT=1)]
        x = np.array([0, 1, 1], dtype=float)
        result = check_mut_mdt_feasibility(x, gen, T=3, initial_state=[0])
        assert result["feasible"] is True
        assert result["violations"] == []


class TestRealWorldRegression:
    """
    The exact schedule that motivated building this function in the
    first place -- previously reported as "feasible" based only on
    power balance, actually MUT-infeasible for two generators. See
    docs/research-log.md.
    """

    def test_catches_the_real_regression_found_in_investigation(self):
        import math

        def make_gen(uid, p_max, fuel_price, heat_rate, start_heat_cold, mut_hr, mdt_hr):
            c_fuel = heat_rate * fuel_price / 1000.0
            su_cost = start_heat_cold * fuel_price
            return Generator(uid, p_max, c_fuel, su_cost, 0.0, MUT=math.ceil(mut_hr), MDT=math.ceil(mdt_hr))

        generators = [
            make_gen("223_STEAM_3", 350.0, 2.11399, 11229, 17384.1, 24, 48),
            make_gen("101_STEAM_3", 76.0,  2.11399, 13270, 5284.8,  8,  4),
            make_gen("107_CC_1",    355.0, 3.88722, 7222,  7215.1,  8,  4.5),
            make_gen("315_CT_7",    55.0,  3.88722, 10342, 1457.4,  2.2, 2.2),
            make_gen("101_CT_1",    20.0,  10.3494, 13114, 5.0,     1,  1),
        ]
        schedule = np.array([
            1, 1, 1, 1,   # 223_STEAM_3
            0, 0, 1, 1,   # 101_STEAM_3
            0, 1, 1, 0,   # 107_CC_1
            0, 0, 1, 0,   # 315_CT_7
            0, 0, 1, 0,   # 101_CT_1
        ], dtype=float)

        result = check_mut_mdt_feasibility(schedule, generators, T=4, initial_state=[0] * 5)
        assert result["feasible"] is False
        violated_generators = {v["generator"] for v in result["violations"]}
        assert violated_generators == {"107_CC_1", "315_CT_7"}
