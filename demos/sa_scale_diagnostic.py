"""
FluxQ diagnostic -- energy scale around the best constant schedule.

No SA is run. Prints:
  1. sum(P_max) and demand, to resolve the capacity/demand percentage question
  2. energy of the best constant schedule rebuilt from unit names, compared
     with the value printed by demos/constant_schedule_check.py
  3. dE of every single-bit flip from that schedule (x bits and w bits apart)
  4. T0 as simulated_annealing(seed=23, T0=None) would estimate it, and a
     temperature table over the 250,000-iteration geometric schedule

Save as demos/sa_scale_diagnostic.py and run from the repo root:
    python demos\\sa_scale_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fluxq.data.loader import load_rts_gmlc_generators
from fluxq.data.grid_builder import make_demand_profile
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.solvers.sa_solver import estimate_initial_temperature
from rts_gmlc_scaling_demo import find_gen_csv, build_greedy_feasible_start

N, T = 15, 8
SA_SEED = 23
N_ITER = 250_000
T_MIN = 1e-3  # default of simulated_annealing; used by the scaling demo

BEST_CONST_PRINTED = -11_322_757_836.93
ON_NAMES = [
    "101_CT_1", "101_CT_2", "102_STEAM_3", "102_STEAM_4",
    "107_CC_1", "113_CT_1", "113_CT_2", "115_STEAM_1",
]


def main():
    gens = load_rts_gmlc_generators(find_gen_csv())[:N]
    names = [g.name for g in gens]
    total_cap = sum(g.P_max for g in gens)
    demand = make_demand_profile(T=T, peak_demand=total_cap * 0.85, profile="flat")
    print(f"sum(P_max) over {N} gens = {total_cap:.1f} MW; "
          f"demand[0] = {demand[0]:.1f} MW ({100 * demand[0] / total_cap:.1f}% of that sum)")

    builder = QUBOBuilder(gens, T=T, enable_mut_mdt=True)
    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    lambdas = LambdaConfig(
        lambda_1=lam1 * 1.5, lambda_3=lam_mm, lambda_4=lam_mm, lambda_link=lam_mm * 2
    )
    Q = builder.build(demand, lambdas)
    n, nr, nw = builder.n_vars, builder.n_vars_real, builder.n_vars_w
    print(f"n_vars={n} (real={nr}, w={nw}); lambda_1={lambdas.lambda_1:,.0f}, "
          f"lambda_3={lam_mm:.4g}, lambda_link={lam_mm * 2:.4g}")

    def schedule_from_on_set(on_set):
        # Same layout as build_greedy_feasible_start: x index i*T+t, w index
        # runs i-major over t=1..T-1 with w = x[t-1] * x[t].
        x = np.zeros(nr)
        for i in on_set:
            x[i * T:(i + 1) * T] = 1.0
        w = np.zeros(nw)
        idx = 0
        for i in range(N):
            for t in range(1, T):
                w[idx] = x[i * T + t - 1] * x[i * T + t]
                idx += 1
        return np.concatenate([x, w])

    missing = [nm for nm in ON_NAMES if nm not in names]
    if missing:
        raise SystemExit(f"Unit names not found among the first {N} generators: {missing}")
    best_x = schedule_from_on_set([names.index(nm) for nm in ON_NAMES])
    e_best = builder.evaluate(best_x)
    print(f"\nBest-constant schedule rebuilt from names: energy = {e_best:,.2f}")
    print(f"  printed by constant_schedule_check.py     : {BEST_CONST_PRINTED:,.2f}")
    print(f"  difference                                : {e_best - BEST_CONST_PRINTED:,.4f}")

    def flip_report(label, x0):
        e0 = builder.evaluate(x0)
        d = np.empty(n)
        for k in range(n):
            xk = x0.copy()
            xk[k] = 1.0 - xk[k]
            d[k] = builder.evaluate(xk) - e0
        print(f"\n[{label}] energy = {e0:,.2f}")
        for tag, sl in (("x bits", slice(0, nr)), ("w bits", slice(nr, n))):
            dd = d[sl]
            pos = dd[dd > 0]
            print(f"  {tag}: {dd.size} single flips, {int((dd <= 0).sum())} with dE <= 0")
            if pos.size:
                p10, p50, p90 = np.percentile(pos, [10, 50, 90])
                print(f"    positive dE: min={pos.min():.3e}  p10={p10:.3e}  "
                      f"median={p50:.3e}  p90={p90:.3e}  max={pos.max():.3e}")
        return d

    d_best = flip_report("best constant schedule", best_x)

    greedy_x, greedy_on, greedy_total = build_greedy_feasible_start(
        gens, demand, T, nr, nw, T
    )
    print(f"\nGreedy seed from the scaling demo: {len(greedy_on)} units, "
          f"{greedy_total:.1f} MW vs demand {demand[0]:.1f} MW")
    flip_report("greedy seed", greedy_x)

    T0 = estimate_initial_temperature(Q, n, seed=SA_SEED)
    print(f"\nestimate_initial_temperature(seed={SA_SEED}) = {T0:.4e}   T_min = {T_MIN:g}")

    pos_x = d_best[:nr][d_best[:nr] > 0]
    if pos_x.size == 0:
        print("No positive x-flip dE from the best constant schedule; table skipped.")
        return
    ref_min, ref_med = float(pos_x.min()), float(np.median(pos_x))
    print(f"Reference dE from best constant schedule (x flips): "
          f"min={ref_min:.3e}, median={ref_med:.3e}")
    print(f"\n{'frac':>5} {'iteration':>10} {'T':>12} {'exp(-min/T)':>13} {'exp(-med/T)':>13}")
    for f in np.linspace(0.0, 1.0, 11):
        Tf = T0 * (T_MIN / T0) ** f
        print(f"{f:5.1f} {int(f * N_ITER):10d} {Tf:12.3e} "
              f"{np.exp(-ref_min / Tf):13.3e} {np.exp(-ref_med / Tf):13.3e}")


if __name__ == "__main__":
    main()