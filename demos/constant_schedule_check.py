"""
Enumerate all 2^15 constant schedules (each unit ON or OFF for all T steps),
with each w set to its true product, and compare the best to SA's energy.

best_const < SA  -> SA missed a feasible, lower-energy point (search side).
best_const >= SA -> SA is at least as good as any constant schedule; then
                    look at the formulation / penalty weights instead.
A constant schedule is a feasible point, so its energy is an upper bound on
the true minimum, never a proof of optimality.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.loader import load_rts_gmlc_generators
from fluxq.data.grid_builder import make_demand_profile
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.validator import power_imbalance, check_mut_mdt_feasibility

SA_ENERGY = -11_321_450_240.87   # must come from the SAME Q as built below


def find_gen_csv() -> Path:
    env = os.environ.get("RTS_GMLC_PATH")
    if env and Path(env).exists():
        return Path(env)
    p = Path(__file__).resolve().parent.parent.parent / "RTS-GMLC" / "RTS_Data" / "SourceData" / "gen.csv"
    if p.exists():
        return p
    raise FileNotFoundError("RTS-GMLC gen.csv not found (set RTS_GMLC_PATH).")


def main():
    N, T = 15, 8
    gens = load_rts_gmlc_generators(find_gen_csv())[:N]
    cap = sum(g.P_max for g in gens)
    demand = make_demand_profile(T=T, peak_demand=cap * 0.85, profile="flat")

    builder = QUBOBuilder(gens, T=T, enable_mut_mdt=True)
    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    lambdas = LambdaConfig(lambda_1=lam1 * 1.5, lambda_3=lam_mm,
                           lambda_4=lam_mm, lambda_link=lam_mm * 2)
    Q = builder.build(demand, lambdas)
    n_real, n_w, n_vars = builder.n_vars_real, builder.n_vars_w, builder.n_vars
    assert n_real == N * T and n_w == N * (T - 1) and Q.shape == (n_vars, n_vars)

    print(f"suggest_lambda()={lam1:,.4g}  suggest_lambda_mut_mdt()={lam_mm:,.4g}")
    print(f"lambda_1={lambdas.lambda_1:,.4g} lambda_3={lambdas.lambda_3:,.4g} "
          f"lambda_4={lambdas.lambda_4:,.4g} lambda_link={lambdas.lambda_link:,.4g}")
    print(f"n_vars={n_vars}  demand={demand[0]:.1f} MW  capacity={cap:.1f} MW")

    def make_X(masks):
        bits = ((masks[:, None] >> np.arange(N)) & 1).astype(np.float64)
        X = np.zeros((len(masks), n_vars))
        X[:, :n_real] = np.repeat(bits, T, axis=1)        # column i*T+t = x_i
        X[:, n_real:] = np.repeat(bits, T - 1, axis=1)    # w = x_{t-1}*x_t = x_i
        return X

    def energy(X):
        return np.einsum("bi,bi->b", X @ Q, X)

    # Sanity check against the builder's own evaluate()
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 2**N, size=5).astype(np.uint32)
    Xp, Ep = make_X(probe), energy(make_X(probe))
    for k in range(len(probe)):
        ref = builder.evaluate(Xp[k])
        assert np.isclose(Ep[k], ref, rtol=1e-9, atol=1e-6), (Ep[k], ref)
    print("sanity: vectorised energy matches builder.evaluate on 5 random schedules")

    top = []
    for start in range(0, 2**N, 4096):
        masks = np.arange(start, min(start + 4096, 2**N), dtype=np.uint32)
        E = energy(make_X(masks))
        for k in np.argsort(E)[:5]:
            top.append((float(E[k]), int(masks[k])))
        top = sorted(top)[:5]

    best_e, best_mask = top[0]
    x = make_X(np.array([best_mask], dtype=np.uint32))[0]
    on = [i for i in range(N) if (best_mask >> i) & 1]

    print(f"\nbest constant schedule : {best_e:,.2f}")
    print(f"SA energy              : {SA_ENERGY:,.2f}")
    print(f"best_const - SA        : {best_e - SA_ENERGY:,.2f}  (negative => SA missed it)")
    print(f"units ON               : {[gens[i].name for i in on]}")
    print(f"ON capacity            : {sum(gens[i].P_max for i in on):.1f} MW")
    print("runner-up energies     :", [f"{e:,.0f}" for e, _ in top[1:]])

    imb = power_imbalance(x, gens, demand, T)
    print(f"max |imbalance|        : {float(np.max(np.abs(imb))):.4f} MW")
    feas = check_mut_mdt_feasibility(x, gens, T, initial_state=[0] * N)
    print(f"MUT/MDT feasible       : {feas['feasible']}  (violations: {len(feas['violations'])})")
    print(f"costs                  : {builder.costs(x)}")


if __name__ == "__main__":
    main()