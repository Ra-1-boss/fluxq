"""
FluxQ w-soundness check (no SA).

Q has no w-w off-diagonal entries (constraints.py), so for a fixed real
schedule x the energy is E(x, w) = E(x, 0) + sum_k c_k w_k, where
    c_k = Q[w_k, w_k] + sum_j Q[j, w_k] * x_j      (j over real variables).
The best w_k is therefore 1 iff c_k < 0. The formulation is "sound" if that
best w always equals the true product x[t-1]*x[t].

Prints, for random and sparse-switching schedules and several lambda_link
multiples (of lambda_3 = lambda_4 = suggest_lambda_mut_mdt()):
  - how often w prefers != product (cheat bits: product 0 but c<0;
    stuck bits: product 1 but c>0), and the largest energy gain from cheating,
    in units of lambda_3.
Also lists, for the best constant schedule at lambda_link = 2*lambda_3, which w
bits have product 0 and c <= 0 (ties or cheats).

Save as demos/w_soundness_check.py; run from the repo root:
    python demos\\w_soundness_check.py
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
from rts_gmlc_scaling_demo import find_gen_csv

N, T = 15, 8
S = 20_000
LINK_MULTIPLES = [2.0, 4.0, 7.0, 10.0]
ON_NAMES = [
    "101_CT_1", "101_CT_2", "102_STEAM_3", "102_STEAM_4",
    "107_CC_1", "113_CT_1", "113_CT_2", "115_STEAM_1",
]


def main():
    gens = load_rts_gmlc_generators(find_gen_csv())[:N]
    names = [g.name for g in gens]
    cap = sum(g.P_max for g in gens)
    demand = make_demand_profile(T=T, peak_demand=cap * 0.85, profile="flat")
    builder = QUBOBuilder(gens, T=T, enable_mut_mdt=True)
    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    nr, nw, n = builder.n_vars_real, builder.n_vars_w, builder.n_vars
    tol = 1e-6 * lam_mm
    print(f"lambda_3=lambda_4={lam_mm:.4g}; MUT/MDT of the {N} units:")
    print("  ", [(g.name, g.MUT, g.MDT) for g in gens])

    rng = np.random.default_rng(0)
    uniform = rng.integers(0, 2, size=(S, nr)).astype(np.float64)
    start = rng.integers(0, 2, size=(S, N, 1))
    sw = (rng.random((S, N, T)) < 0.2).astype(int)
    sw[:, :, 0] = 0
    sparse = ((start + np.cumsum(sw, axis=2)) % 2).reshape(S, nr).astype(np.float64)
    families = {"uniform random": uniform, "sparse switching": sparse}

    def products(X):
        xr = X.reshape(-1, N, T)
        return (xr[:, :, :-1] * xr[:, :, 1:]).reshape(len(X), N * (T - 1))

    def coeffs(Q, X):
        return np.diag(Q)[nr:][None, :] + X @ Q[:nr, nr:]

    for mult in LINK_MULTIPLES:
        lambdas = LambdaConfig(lambda_1=lam1 * 1.5, lambda_3=lam_mm,
                               lambda_4=lam_mm, lambda_link=lam_mm * mult)
        Q = builder.build(demand, lambdas)

        # structure + identity checks
        n_ww = int(np.count_nonzero(np.triu(Q[nr:, nr:], 1)))
        xs = uniform[:3]
        for k in range(3):
            ws = rng.integers(0, 2, size=nw).astype(np.float64)
            full = np.concatenate([xs[k], ws])
            full0 = np.concatenate([xs[k], np.zeros(nw)])
            lhs = float(full @ Q @ full - full0 @ Q @ full0)
            rhs = float(coeffs(Q, xs[k:k + 1])[0] @ ws)
            assert np.isclose(lhs, rhs, rtol=1e-9, atol=1e-3), (lhs, rhs)
        print(f"\n=== lambda_link = {mult:g} x lambda_3 ({lambdas.lambda_link:.4g}) ; "
              f"w-w off-diagonal nonzeros: {n_ww} ; separability identity OK ===")

        for fam, X in families.items():
            c = coeffs(Q, X)
            prod = products(X)
            cheat = (prod == 0) & (c < -tol)
            tie0 = (prod == 0) & (np.abs(c) <= tol)
            stuck = (prod == 1) & (c > tol)
            gain = np.where(cheat, -c, 0.0)
            print(f"  {fam:<17} cheat bits {100 * cheat.mean():6.3f}% | "
                  f"samples with any cheat {100 * cheat.any(axis=1).mean():6.2f}% | "
                  f"stuck bits {100 * stuck.mean():6.3f}% | zero-tie bits {100 * tie0.mean():6.3f}% | "
                  f"max single gain {gain.max() / lam_mm:5.2f} x lambda_3 | "
                  f"mean total gain/sample {gain.sum(axis=1).mean() / lam_mm:6.2f} x lambda_3")

    # best constant schedule at lambda_link = 2 * lambda_3
    lambdas = LambdaConfig(lambda_1=lam1 * 1.5, lambda_3=lam_mm,
                           lambda_4=lam_mm, lambda_link=lam_mm * 2)
    Q = builder.build(demand, lambdas)
    x = np.zeros((1, nr))
    for nm in ON_NAMES:
        i = names.index(nm)
        x[0, i * T:(i + 1) * T] = 1.0
    c = coeffs(Q, x)[0]
    prod = products(x)[0]
    print("\nBest constant schedule, lambda_link = 2 x lambda_3: w bits with product 0 and c <= tol")
    found = False
    for k in range(nw):
        if prod[k] == 0 and c[k] <= tol:
            i, t = divmod(k, T - 1)
            g = gens[i]
            print(f"  {g.name:<14} t={t + 1}  MUT={g.MUT} MDT={g.MDT}  c = {c[k]:.4g} "
                  f"({c[k] / lam_mm:+.3f} x lambda_3)")
            found = True
    if not found:
        print("  none")


if __name__ == "__main__":
    main()