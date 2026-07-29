"""
FluxQ Real-Data Demo
====================
5 Real RTS-GMLC Generators, 4-Timestep Horizon -- Brute-Force Verification

This is the first time FluxQ's QUBO builder runs against real generator
data instead of the synthetic toy system. All 5 generators are drawn
directly from RTS-GMLC (github.com/GridMod/RTS-GMLC), a real, published
grid test system -- see fluxq/data/grid_builder.py::make_rts_gmlc_sample()
for the exact source values and derivation formulas.

Same verification method as Phase 0's toy_brute_force.py: enumerate all
2^20 = 1,048,576 states, find the true minimum, and confirm it matches
what hand-reasoning about the merit order predicts.

Usage
-----
    python demos/rts_gmlc_5gen_brute_force.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import make_rts_gmlc_sample
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.validator import decode_solution, power_imbalance


# ---------------------------------------------------------------------------
# Brute-force solver (same as Phase 0 -- n_vars=20 is safely enumerable)
# ---------------------------------------------------------------------------

def brute_force(Q: np.ndarray, n_vars: int, top_k: int = 8):
    n_states = 1 << n_vars
    print(f"  Enumerating {n_states:,} states for {n_vars} binary variables...")
    t0 = time.perf_counter()

    indices = np.arange(n_states, dtype=np.uint32)
    X = ((indices[:, None] >> np.arange(n_vars, dtype=np.uint32)[None, :]) & 1).astype(np.float64)
    QX = X @ Q
    energies = np.einsum("si,si->s", QX, X)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s.")

    order = np.argsort(energies)
    return [(float(energies[s]), X[s].copy()) for s in order[:top_k]]


def main():
    print()
    print("=" * 70)
    print("  FluxQ -- First Real-Data Verification")
    print("  5 real RTS-GMLC generators x 4 timesteps -> 2^20 states")
    print("=" * 70)

    generators = make_rts_gmlc_sample()
    T = 4
    dt = 1.0

    print()
    print("  Generators (source: RTS-GMLC gen.csv)")
    print(f"  {'Name':<14} {'P_max':>6}  {'c_fuel':>8}  {'SU_$':>10}  {'MUT':>4}  {'MDT':>4}")
    print("  " + "-" * 60)
    for g in generators:
        print(f"  {g.name:<14} {g.P_max:>6.0f}  {g.c_fuel:>7.2f}$  {g.SU_cost:>9.2f}$  {g.MUT:>4}  {g.MDT:>4}")

    # Demand profile designed so each timestep has exactly one exact-match
    # subset of generator capacities -- verified by hand across all 32
    # possible subsets of {350, 76, 355, 55, 20} before writing this script.
    demand = np.array([350.0, 705.0, 856.0, 426.0])

    print()
    print(f"  Demand profile: {demand} MW")
    print(f"  Total capacity: {sum(g.P_max for g in generators):.0f} MW")
    print()
    print("  Hand-verified unique exact-match subset per timestep:")
    print("    t=0 (350 MW): {223_STEAM_3}")
    print("    t=1 (705 MW): {223_STEAM_3, 107_CC_1}")
    print("    t=2 (856 MW): {all five}")
    print("    t=3 (426 MW): {223_STEAM_3, 101_STEAM_3}")

    builder = QUBOBuilder(generators, T=T, dt=dt)
    lam = builder.suggest_lambda()
    print()
    print(f"  suggest_lambda(): {lam:.5f}")

    Q = builder.build(demand, LambdaConfig(lambda_1=lam * 1.5))
    builder.print_info()

    print()
    print("  Brute-force search")
    print("  " + "-" * 40)
    results = brute_force(Q, builder.n_vars, top_k=8)

    best_energy, best_x = results[0]
    decoded = decode_solution(best_x, generators, T)
    schedule = decoded["schedule"]
    imbal = power_imbalance(best_x, generators, demand, T)

    print()
    print("  Best schedule found (1=ON, 0=OFF):")
    header = f"  {'Generator':<14}" + "  ".join(f"t{t}" for t in range(T))
    print(header)
    for i, g in enumerate(generators):
        row = "   ".join(str(schedule[i, t]) for t in range(T))
        print(f"  {g.name:<14}{row}")

    print()
    print(f"  Max power imbalance across all timesteps: {np.max(np.abs(imbal)):.2f} MW")

    costs = builder.costs(best_x)
    print()
    print(f"  Total real cost breakdown:")
    print(f"    Fuel:     ${costs['fuel']:>12,.2f}")
    print(f"    Startup:  ${costs['startup']:>12,.2f}")
    print(f"    Shutdown: ${costs['shutdown']:>12,.2f}")
    print(f"    TOTAL:    ${costs['total']:>12,.2f}")

    print()
    startup_share = costs['startup'] / costs['total'] * 100
    print(f"  Startup cost is {startup_share:.1f}% of total cost over this 4-hour window.")
    print(f"  (This is a genuine property of real generator economics -- large")
    print(f"   thermal units have startup costs that dwarf a short-horizon fuel")
    print(f"   bill, which is exactly why MUT/MDT-aware scheduling matters.)")

    # ── Sanity check against hand-reasoned expectation ─────────────────
    print()
    print("  Checking against hand-reasoned expectation...")
    print("  Expected: 223_STEAM_3 stays ON continuously (required every timestep,")
    print("  paying its startup cost exactly once).")

    steam3_idx = [g.name for g in generators].index("223_STEAM_3")
    steam3_row = schedule[steam3_idx]
    if all(steam3_row == 1):
        print("  ✓ Confirmed: 223_STEAM_3 is ON at every timestep in the true optimum.")
    else:
        print(f"  ✗ Unexpected: 223_STEAM_3 schedule is {list(steam3_row)} -- hand reasoning was wrong somewhere, worth investigating.")

    assert np.max(np.abs(imbal)) < 1e-6, "Power balance should be exactly satisfied at the true optimum"
    print("  ✓ Power balance exactly satisfied (zero imbalance) at every timestep.")

    print()
    print("=" * 70)
    print("  First real-data verification complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
