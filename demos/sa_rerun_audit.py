"""
FluxQ audit -- rerun the 250k-iteration SA on the SAME Q used by
demos/constant_schedule_check.py and audit the result.

Prints, for the best constant schedule and for SA's best_x:
  - energy, power-balance term, link-penalty term, remainder, builder costs
  - every violation reported by check_mut_mdt_feasibility, with the label the
    demo's classify_violations would give it (required >= T -> "horizon-limited")
  - a per-unit table (Pmax, MUT, MDT, schedule row)
For SA it also prints T0_used, the spread of per-restart best energies, and
the difference from the energy quoted in constant_schedule_check.py.

Save as demos/sa_rerun_audit.py and run from the repo root:
    python demos\\sa_rerun_audit.py
Runtime is about that of the earlier 250,000 x 15 run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fluxq.data.loader import load_rts_gmlc_generators
from fluxq.data.grid_builder import make_demand_profile
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.validator import power_imbalance, check_mut_mdt_feasibility
from fluxq.solvers.sa_solver import simulated_annealing
from rts_gmlc_scaling_demo import find_gen_csv

N, T = 15, 8
SA_SEED = 23
N_ITER = 250_000
N_RESTARTS = 15
SA_ENERGY_QUOTED = -11_321_450_240.87        # value hard-coded in constant_schedule_check.py
BEST_CONST_PRINTED = -11_322_757_836.93
ON_NAMES = [
    "101_CT_1", "101_CT_2", "102_STEAM_3", "102_STEAM_4",
    "107_CC_1", "113_CT_1", "113_CT_2", "115_STEAM_1",
]


def main():
    gens = load_rts_gmlc_generators(find_gen_csv())[:N]
    names = [g.name for g in gens]
    by_name = {g.name: g for g in gens}
    cap = sum(g.P_max for g in gens)
    demand = make_demand_profile(T=T, peak_demand=cap * 0.85, profile="flat")

    builder = QUBOBuilder(gens, T=T, enable_mut_mdt=True)
    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    lambdas = LambdaConfig(
        lambda_1=lam1 * 1.5, lambda_3=lam_mm, lambda_4=lam_mm, lambda_link=builder.suggest_lambda_link(lam_mm, lam_mm)
    )
    Q = builder.build(demand, lambdas)
    n, nr, nw = builder.n_vars, builder.n_vars_real, builder.n_vars_w
    offset = lambdas.lambda_1 * float(np.sum(np.asarray(demand, dtype=float) ** 2))
    print(f"n_vars={n}  demand={demand[0]:.2f} MW  sum(P_max)={cap:.1f} MW")
    print(f"lambda_1={lambdas.lambda_1:,.1f}  lambda_3=lambda_4={lam_mm:.4g}  "
          f"lambda_link={lambdas.lambda_link:.4g}")
    print(f"dropped constant  lambda_1 * sum(D_t^2) = {offset:,.2f}")

    # --- best constant schedule, rebuilt from names (same layout as the check) ---
    missing = [nm for nm in ON_NAMES if nm not in names]
    if missing:
        raise SystemExit(f"Unit names not found: {missing}")
    best_x = np.zeros(n)
    for nm in ON_NAMES:
        i = names.index(nm)
        best_x[i * T:(i + 1) * T] = 1.0
        best_x[nr + i * (T - 1): nr + (i + 1) * (T - 1)] = 1.0   # w = x_{t-1}*x_t = 1

    def decompose(x):
        e = builder.evaluate(x)
        imb = power_imbalance(x, gens, demand, T)
        balance = lambdas.lambda_1 * float(np.sum(np.asarray(imb) ** 2))
        xr = x[:nr].reshape(N, T)
        w = x[nr:].reshape(N, T - 1)        # index nr + i*(T-1) + (t-1)
        a, b = xr[:, :-1], xr[:, 1:]
        link = lambdas.lambda_link * float(np.sum(a * b - 2 * a * w - 2 * b * w + 3 * w))
        n_link_bad = int(np.sum(w != a * b))
        remainder = e + offset - balance - link
        return dict(energy=e, balance=balance, link=link, n_link_bad=n_link_bad,
                    remainder=remainder, max_imb=float(np.max(np.abs(imb))))

    def show(label, x):
        d = decompose(x)
        costs = builder.costs(x)
        print(f"\n[{label}]")
        print(f"  Q energy (builder.evaluate)   : {d['energy']:,.2f}")
        print(f"  lambda_1 * sum (P-D)^2         : {d['balance']:,.2f}   "
              f"(max |imbalance| {d['max_imb']:.4f} MW)")
        print(f"  link penalty (hand formula)   : {d['link']:,.2f}   "
              f"({d['n_link_bad']} of {nw} w bits differ from x[t-1]*x[t])")
        print(f"  remainder = E + offset - balance - link : {d['remainder']:,.2f}")
        print(f"  builder.costs total           : {costs['total']:,.2f}")
        print(f"  remainder - costs total       : {d['remainder'] - costs['total']:,.2f}"
              f"   (= MUT/MDT penalty only if objective.py matches builder.costs; unverified)")

        feas = check_mut_mdt_feasibility(x, gens, T, initial_state=[0] * N)
        viols = feas["violations"]
        n_hl = 0
        print(f"  validator: feasible={feas['feasible']}, violations={len(viols)}")
        for v in viols:
            g = by_name[v["generator"]]
            req = g.MUT if v["type"] == "MUT" else g.MDT
            label_demo = "horizon-limited" if req >= T else "genuine"
            n_hl += label_demo == "horizon-limited"
            print(f"    [demo label: {label_demo}] {v['message']}")
        print(f"  demo classifier would report {len(viols) - n_hl} genuine, {n_hl} horizon-limited")

        sched = x[:nr].reshape(N, T).astype(int)
        viol_units = {v["generator"] for v in viols}
        print(f"  {'unit':<16}{'Pmax':>8}{'MUT':>5}{'MDT':>5}  t0..t{T - 1}   (* = has a violation)")
        for i, g in enumerate(gens):
            mark = "*" if g.name in viol_units else " "
            row = "".join(str(v) for v in sched[i])
            print(f"{mark} {g.name:<16}{g.P_max:8.1f}{g.MUT:5d}{g.MDT:5d}  {row}")

    e_rebuilt = builder.evaluate(best_x)
    print(f"\nrebuilt best-constant energy {e_rebuilt:,.2f} vs printed "
          f"{BEST_CONST_PRINTED:,.2f}  (diff {e_rebuilt - BEST_CONST_PRINTED:,.4f})")
    show("best constant schedule", best_x)

    print(f"\nRunning SA: {N_RESTARTS} restarts x {N_ITER:,} iterations, seed {SA_SEED}, "
          f"random start, default T0/T_min ...")
    t0 = time.perf_counter()
    res = simulated_annealing(Q, n, n_iterations=N_ITER, n_restarts=N_RESTARTS,
                          seed=SA_SEED, T0=1e7, T_min=1e4)
    print(f"Done in {time.perf_counter() - t0:.1f}s.")
    print(f"T0_used                    : {res['T0_used']:.6e}")
    print(f"best_energy (solver)       : {res['best_energy']:,.2f}")
    print(f"builder.evaluate(best_x)   : {builder.evaluate(res['best_x']):,.2f}")
    print(f"quoted in check script     : {SA_ENERGY_QUOTED:,.2f}  "
          f"(difference {res['best_energy'] - SA_ENERGY_QUOTED:,.2f})")
    print("per-restart best energies  :", [f"{e:,.0f}" for e in sorted(res["all_energies"])])
    print(f"best constant - this SA    : {builder.evaluate(best_x) - res['best_energy']:,.2f}"
          f"  (negative => SA missed it)")
    show("SA best_x", res["best_x"])


if __name__ == "__main__":
    main()