"""
FluxQ Scaling Demo -- Seeded-Start Experiment
===============================================
Third variant in an ongoing investigation -- see docs/research-log.md.

Two prior tests, each changing exactly one variable:
  1. Iteration budget: 60,000 -> 250,000 (15 restarts). Genuine MUT/MDT
     violations dropped 19 -> 11. Real improvement, not full resolution.
  2. Demand shape: 'daily' (oscillating) -> 'flat' (constant), same
     250,000-iteration budget. Power imbalance dropped sharply
     (5.27 -> 1.56 MW) but genuine violations barely moved (11 -> 10).
     This RULES OUT demand shape as the primary driver of the remaining
     violations, even though it clearly mattered for power balance.

This third test targets a different hypothesis: under FLAT demand, the
lowest-cost feasible answer is almost certainly "commit one fixed,
sufficient subset of generators, turn them on once, leave them on for
the entire horizon, never cycle" -- there is no cost reason to ever
switch anything under constant demand. If SA is still finding spurious
single-timestep flickers (e.g. "started at t=0, off again at t=1") under
these easy conditions, that looks more like the random-start search
genuinely not settling than a structural infeasibility.

This test seeds SA from a hand-constructed, trivially MUT/MDT-feasible
starting point -- one fixed generator subset, greedily chosen to meet
demand, held on for the whole horizon -- and checks whether the search
holds onto that stability (few or no genuine violations) or wanders away
from it regardless (pointing at the move set / cooling schedule itself,
not the starting point).

Usage
-----
    python demos/rts_gmlc_scaling_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.loader import load_rts_gmlc_generators
from fluxq.data.grid_builder import make_demand_profile
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.validator import decode_solution, power_imbalance, check_mut_mdt_feasibility
from fluxq.solvers.sa_solver import simulated_annealing


def find_gen_csv() -> Path:
    import os
    env_path = os.environ.get("RTS_GMLC_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent.parent
    default_path = repo_root.parent / "RTS-GMLC" / "RTS_Data" / "SourceData" / "gen.csv"
    if default_path.exists():
        return default_path
    raise FileNotFoundError(
        "RTS-GMLC gen.csv not found. Clone github.com/GridMod/RTS-GMLC as a "
        "sibling of this repo, or set RTS_GMLC_PATH."
    )


def classify_violations(violations, generators, T):
    horizon_limited, genuine = [], []
    gen_by_name = {g.name: g for g in generators}
    for v in violations:
        gen = gen_by_name[v["generator"]]
        required = gen.MUT if v["type"] == "MUT" else gen.MDT
        if required >= T:
            horizon_limited.append(v)
        else:
            genuine.append(v)
    return horizon_limited, genuine


def build_greedy_feasible_start(generators, demand, T, n_vars_real, n_vars_w, T_horizon):
    """
    Construct a trivially MUT/MDT-feasible starting schedule: greedily
    pick generators (in the order given) until their combined capacity
    meets or exceeds the flat demand level, then hold that exact set ON
    for the entire horizon and everyone else OFF for the entire horizon.

    Because every committed generator runs continuously for the whole
    horizon, no MUT/MDT check can ever fail for it (it never shuts down
    within the visible window), and no idle generator ever starts. This
    is not necessarily cost-optimal -- it's a deliberately safe,
    structurally feasible point to seed the search from, to test whether
    SA holds onto that feasibility or wanders away from it regardless.

    Returns the FULL x vector (real commitment variables plus correctly
    computed auxiliary w variables), ready to pass as initial_x.
    """
    N = len(generators)
    target = float(demand[0])  # flat demand -- every timestep is the same
    on_set = []
    total = 0.0
    for i, gen in enumerate(generators):
        if total >= target:
            break
        on_set.append(i)
        total += gen.P_max

    real_x = np.zeros(n_vars_real, dtype=np.float64)
    for i in on_set:
        for t in range(T_horizon):
            real_x[i * T_horizon + t] = 1.0

    w = np.zeros(n_vars_w, dtype=np.float64)
    idx = 0
    for i in range(N):
        for t in range(1, T_horizon):
            a = real_x[i * T_horizon + (t - 1)]
            b = real_x[i * T_horizon + t]
            w[idx] = a * b
            idx += 1

    return np.concatenate([real_x, w]), on_set, total


def main():
    print()
    print("=" * 70)
    print("  FluxQ Scaling Demo -- Seeded-Start Experiment")
    print("=" * 70)

    gen_csv = find_gen_csv()
    all_dispatchable = load_rts_gmlc_generators(gen_csv)
    print(f"\n  Loaded {len(all_dispatchable)} real dispatchable generators from {gen_csv.name}")

    N = 15
    T = 8
    generators = all_dispatchable[:N]
    total_capacity = sum(g.P_max for g in generators)

    print(f"\n  --- N={N} real generators, T={T} timesteps, FLAT demand ---")
    demand = make_demand_profile(T=T, peak_demand=total_capacity * 0.85, profile="flat")
    print(f"  Demand: {np.round(demand).astype(int)} MW  (constant, {100*demand[0]/total_capacity:.1f}% of capacity)")

    builder = QUBOBuilder(generators, T=T, enable_mut_mdt=True)
    print(f"  n_vars: {builder.n_vars}")

    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    lambdas = LambdaConfig(lambda_1=lam1 * 1.5, lambda_3=lam_mm, lambda_4=lam_mm, lambda_link=lam_mm * 2)
    Q = builder.build(demand, lambdas)

    seed_x, on_set, total = build_greedy_feasible_start(
        generators, demand, T, builder.n_vars_real, builder.n_vars_w, T
    )
    print(f"\n  Trivially-feasible seed: {len(on_set)} generators held ON the whole horizon")
    print(f"    {[generators[i].name for i in on_set]}")
    print(f"    Combined capacity: {total:.0f} MW vs demand {demand[0]:.0f} MW")
    seed_energy = builder.evaluate(seed_x)
    print(f"    QUBO energy of this seed (before any search): {seed_energy:,.2f}")

    print("\n  Running SA SEEDED from this feasible point (15 restarts, 250,000 iterations)...")
    t0 = time.perf_counter()
    result = simulated_annealing(
        Q, builder.n_vars, n_iterations=250000, n_restarts=15,
        initial_x=seed_x, seed=23,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s.")
    print(f"  Best energy: {result['best_energy']:,.2f}  (seed was {seed_energy:,.2f})")

    imbal = power_imbalance(result["best_x"], generators, demand, T)
    max_imbal = float(np.max(np.abs(imbal)))
    print(f"\n  Power balance -- max imbalance: {max_imbal:.4f} MW", "(exact)" if max_imbal < 1e-6 else "(NOT exact)")

    feas = check_mut_mdt_feasibility(result["best_x"], generators, T, initial_state=[0] * N)
    horizon_limited, genuine = classify_violations(feas["violations"], generators, T)
    print(f"\n  MUT/MDT feasibility:")
    print(f"    Feasible: {feas['feasible']}")
    print(f"    Genuine violations: {len(genuine)}  (was 10 from a random start, same budget)")
    print(f"    Horizon-limited: {len(horizon_limited)}")
    if genuine:
        print(f"\n    Genuine violations:")
        for v in genuine:
            print(f"      - {v['message']}")

    costs = builder.costs(result["best_x"])
    print(f"\n  Total cost: ${costs['total']:,.2f}")

    print()
    print("=" * 70)
    print("  Seeded-start experiment complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
