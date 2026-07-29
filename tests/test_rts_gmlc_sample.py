"""
Tests for fluxq.data.grid_builder.make_rts_gmlc_sample().

These tests verify that the derived c_fuel, SU_cost, MUT, and MDT values
for each of the 5 real RTS-GMLC generators match hand-computed values,
worked out independently from the raw gen.csv source fields before any
code was written. This is the same discipline used throughout Phase 0:
verify by hand first, then confirm the code produces the same answer.

Run with:
    python -m pytest tests/test_rts_gmlc_sample.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import make_rts_gmlc_sample


# ---------------------------------------------------------------------------
# Hand-derived expected values
# ---------------------------------------------------------------------------
# Recomputed carefully by hand from raw gen.csv fields:
#   c_fuel  = HR_avg_0 * Fuel Price / 1000
#   SU_cost = Start Heat Cold * Fuel Price + Non Fuel Start Cost
#
# Tolerance is 0.05 (5 cents / 5 cents-per-MWh) to allow for minor
# floating-point rounding differences between hand arithmetic and the
# code's own computation -- not to hide a real discrepancy.

EXPECTED = {
    "223_STEAM_3": {"P_max": 350.0, "c_fuel": 23.738, "SU_cost": 36749.81, "SD_cost": 0.0, "MUT": 24, "MDT": 48},
    "101_STEAM_3": {"P_max": 76.0,  "c_fuel": 28.053, "SU_cost": 11172.01, "SD_cost": 0.0, "MUT": 8,  "MDT": 4},
    "107_CC_1":    {"P_max": 355.0, "c_fuel": 28.074, "SU_cost": 28046.68, "SD_cost": 0.0, "MUT": 8,  "MDT": 5},
    "315_CT_7":    {"P_max": 55.0,  "c_fuel": 40.202, "SU_cost": 5665.23,  "SD_cost": 0.0, "MUT": 3,  "MDT": 3},
    "101_CT_1":    {"P_max": 20.0,  "c_fuel": 135.722, "SU_cost": 51.75,   "SD_cost": 0.0, "MUT": 1,  "MDT": 1},
}


class TestRTSGMLCSample:
    def test_returns_five_generators(self):
        """The sample should contain exactly the 5 hand-selected generators."""
        gens = make_rts_gmlc_sample()
        assert len(gens) == 5
        assert {g.name for g in gens} == set(EXPECTED.keys())

    def test_generator_values_match_hand_derivation(self):
        """
        Each generator's derived fields must match hand-computed values
        within a small tolerance.
        """
        gens = {g.name: g for g in make_rts_gmlc_sample()}
        for uid, expected in EXPECTED.items():
            gen = gens[uid]
            assert abs(gen.P_max - expected["P_max"]) < 1e-9, (
                f"{uid}: P_max mismatch"
            )
            assert abs(gen.c_fuel - expected["c_fuel"]) < 0.05, (
                f"{uid}: c_fuel expected ~{expected['c_fuel']}, got {gen.c_fuel:.4f}"
            )
            assert abs(gen.SU_cost - expected["SU_cost"]) < 0.05, (
                f"{uid}: SU_cost expected ~{expected['SU_cost']}, got {gen.SU_cost:.4f}"
            )
            assert abs(gen.SD_cost - expected["SD_cost"]) < 1e-9, (
                f"{uid}: SD_cost mismatch"
            )
            assert gen.MUT == expected["MUT"], (
                f"{uid}: MUT expected {expected['MUT']}, got {gen.MUT}"
            )
            assert gen.MDT == expected["MDT"], (
                f"{uid}: MDT expected {expected['MDT']}, got {gen.MDT}"
            )

    def test_total_capacity(self):
        """Total capacity of the 5-generator sample should be 856 MW."""
        gens = make_rts_gmlc_sample()
        total = sum(g.P_max for g in gens)
        assert abs(total - 856.0) < 1e-9

    def test_merit_order_matches_flexibility_tradeoff(self):
        """
        Cheapest generator (223_STEAM_3) should also have the longest MUT/MDT
        (slowest to cycle). Most expensive (101_CT_1) should have the shortest
        MUT/MDT (fastest to cycle). This is a real property of the selected
        data, not an artifact -- worth protecting with a test since it's the
        whole reason this particular slice was chosen.
        """
        gens = {g.name: g for g in make_rts_gmlc_sample()}
        cheapest = gens["223_STEAM_3"]
        most_expensive = gens["101_CT_1"]

        assert cheapest.c_fuel < most_expensive.c_fuel
        assert cheapest.MUT > most_expensive.MUT
        assert cheapest.MDT > most_expensive.MDT

    def test_generators_are_valid(self):
        """Every generator must pass its own __post_init__ validation."""
        # If any generator were invalid, make_rts_gmlc_sample() itself
        # would have already raised during construction. This test just
        # documents that expectation explicitly.
        gens = make_rts_gmlc_sample()
        assert len(gens) == 5
        for g in gens:
            assert g.P_max > 0
            assert g.c_fuel >= 0
            assert g.SU_cost >= 0
            assert g.SD_cost >= 0
            assert g.MUT >= 1
            assert g.MDT >= 1
