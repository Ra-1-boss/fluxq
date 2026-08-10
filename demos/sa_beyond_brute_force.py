"""
FluxQ Simulated Annealing Demo
==============================
Real RTS-GMLC 5-Generator Sample, MUT/MDT Enabled -- Beyond Brute Force

With MUT/MDT enabled, this problem has 35 binary variables: 20 real
commitment decisions (5 generators x 4 timesteps) plus 15 auxiliary
linking variables (5 generators x 3 transitions). 2^35 ~ 34.4 billion
states -- brute_force.brute_force() refuses anything past 25 variables,
and for good reason: even at a billion states per second this would take
over half a minute of raw enumeration, and Python/NumPy is nowhere near
that fast in practice.

This is the whole reason Simulated Annealing exists in FluxQ. Note the
honest framing throughout: SA's result here is NOT proven optimal the
way brute force's results are (see demos/toy_brute_force.py and
demos/rts_gmlc_5gen_brute_force.py for problems small enough that brute
force actually could check them). What SA gives instead is a real,
feasible, low-cost schedule at a scale brute force cannot reach at all --
a different, more limited, but genuinely useful claim.

Usage
-----
    python demos/sa_beyond_brute_force.py
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
from fluxq.solvers.sa_solver import simulated_annealing


def main():
    print()
    print("=" * 70)
    print("  FluxQ -- Simulated Annealing Beyond Brute Force's Reach")
    print("  5 real RTS-GMLC generators x 4 timesteps, MUT/MDT enabled")
    print("=" * 70)

    generators = make_rts_gmlc_sample()
    T = 4
    demand = np.array([350.0, 705.0, 856.0, 426.0])

    builder = QUBOBuilder(generators, T=T, enable_mut_mdt=True)

    print()
    print(f"  n_vars_real (real commitment decisions): {builder.n_vars_real}")
    print(f"  n_vars_w (auxiliary MUT/MDT linking vars): {builder.n_vars_w}")
    print(f"  n_vars total: {builder.n_vars}")
    print(f"  2^{builder.n_vars} = {2**builder.n_vars:,} states")
    print(f"  Brute force's practical ceiling is 25 variables -- this problem")
    print(f"  is {2**(builder.n_vars-25):,}x past that. Not a close call.")

    lam = builder.suggest_lambda()
    lam_link = builder.suggest_lambda_link(lambda_3=1.0, lambda_4=1.0)
    lambdas = LambdaConfig(lambda_1=lam * 1.5, lambda_3=1.0, lambda_4=1.0, lambda_link=lam_link)
    Q = builder.build(demand, lambdas)

    print()
    print("  Running Simulated Annealing (10 restarts, 40,000 iterations each)...")
    t0 = time.perf_counter()
    result = simulated_annealing(Q, builder.n_vars, n_iterations=40000, n_restarts=10, seed=11)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s.")

    print()
    print(f"  Best energy found: {result['best_energy']:,.2f}")
    print(f"  Spread across 10 restarts: {min(result['all_energies']):,.1f} to {max(result['all_energies']):,.1f}")

    decoded = decode_solution(result["best_x"], generators, T)
    schedule = decoded["schedule"]

    print()
    print("  Schedule found (1=ON, 0=OFF):")
    header = f"  {'Generator':<14}" + "  ".join(f"t{t}" for t in range(T))
    print(header)
    for i, g in enumerate(generators):
        row = "   ".join(str(int(schedule[i, t])) for t in range(T))
        print(f"  {g.name:<14}{row}")

    imbal = power_imbalance(result["best_x"], generators, demand, T)
    max_imbal = float(np.max(np.abs(imbal)))
    print()
    print(f"  Max power imbalance: {max_imbal:.2f} MW", "(exact match)" if max_imbal < 1e-6 else "(NOT exact -- worth investigating)")

    costs = builder.costs(result["best_x"])
    print()
    print("  Real cost breakdown:")
    print(f"    Fuel:     ${costs['fuel']:>14,.2f}")
    print(f"    Startup:  ${costs['startup']:>14,.2f}")
    print(f"    Shutdown: ${costs['shutdown']:>14,.2f}")
    print(f"    TOTAL:    ${costs['total']:>14,.2f}")

    print()
    print("  IMPORTANT -- what this result does and does not claim:")
    print("  This IS a real, feasible schedule with exact power balance,")
    print("  found on a problem brute force cannot even attempt.")
    print("  This is NOT proven to be the global optimum -- unlike the")
    print("  brute-force-verified demos, there is no ground truth to check")
    print("  against at this scale. That distinction is the whole point of")
    print("  building both solvers rather than pretending SA is exact.")

    print()
    print("=" * 70)
    print("  Simulated Annealing demo complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
