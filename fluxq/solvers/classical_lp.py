"""
Classical MILP baseline for the FluxQ unit-commitment QUBO.

What this solves
----------------
The same energy as the QUBO, restricted to w = x[t-1]*x[t] ("product
schedules"), except that the squared balance penalty is replaced by a
piecewise-linear *lower* approximation (tangent lines):

    s^2 >= 2*a*s - a^2      for every tangent point a.

Everything else (fuel, startup, shutdown, per-pair MUT/MDT penalties with
only fully-observed pairs counted, initial_state handling) is encoded exactly,
using continuous indicator variables that the minimiser pushes to
max(0, .) because all their cost coefficients are >= 0.

What the result means
---------------------
* energy_lower_bound : valid lower bound on the minimum QUBO energy over
  product schedules (solver dual bound; tangents only ever under-estimate).
* energy             : exact QUBO energy of the returned schedule, i.e. an
  upper bound on that minimum.
* The true minimum over product schedules lies in [lower_bound, energy].
* window_certifies   : True when the bracket width is due only to the
  solver's MIP gap, not to tangent resolution (see solve_uc_milp).

What it does NOT show
---------------------
* Nothing about SA in general, only a reference value for this instance.
* The minimum over (x, w) jointly equals the product-schedule minimum only
  if lambda_link is large enough (needs
  lambda_link >= max_i((MUT_i-1)*lambda_3 + (MDT_i-1)*lambda_4); this is a
  derivation from constraints.py, checked by the test, not a measured fact).
* If the solver stops at its time limit, read mip_gap before claiming anything.

Energy bookkeeping: Q drops the constants lambda_1*sum(D^2) and, for units
that start ON, SD_i. The MILP objective keeps them out of the way the same
way: E_Q = shifted objective - lambda_1*sum(D^2) - sum(SD_i for init_i == 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class MilpResult:
    status: int
    message: str
    x: Optional[np.ndarray]                 # flat, index i*T+t (QUBO order)
    energy: Optional[float]                 # exact QUBO energy of x (w = product)
    energy_lower_bound: Optional[float]     # solver dual bound, as QUBO energy
    mip_gap: Optional[float]
    n_tangents: int
    window_mw: float
    window_certifies: Optional[bool]


# ---------------------------------------------------------------------------
# Exact model (pure python; also used to cross-check against the QUBO)
# ---------------------------------------------------------------------------

def _offset(gens, demand, lambdas, init) -> float:
    return float(
        lambdas.lambda_1 * np.sum(np.asarray(demand, float) ** 2)
        + sum(g.SD_cost for g, s in zip(gens, init) if s == 1)
    )


def exact_shifted_objective(x, gens, T, demand, lambdas, dt=1.0, initial_state=None) -> float:
    """Cost + penalties of schedule x with w = product, before dropping constants."""
    N = len(gens)
    init = list(initial_state) if initial_state is not None else [0] * N
    X = np.rint(np.asarray(x, float)[: N * T]).astype(int).reshape(N, T)
    total = 0.0
    for i, g in enumerate(gens):
        prev = init[i]
        mut, mdt = int(g.MUT), int(g.MDT)
        for t in range(T):
            cur = int(X[i, t])
            start = max(0, cur - prev)
            shut = max(0, prev - cur)
            total += g.c_fuel * g.P_max * dt * cur + g.SU_cost * start + g.SD_cost * shut
            if start:
                for k in range(1, mut):
                    if t + k >= T:
                        break
                    total += lambdas.lambda_3 * (1 - X[i, t + k])
            if shut:
                for k in range(1, mdt):
                    if t + k >= T:
                        break
                    total += lambdas.lambda_4 * X[i, t + k]
            prev = cur
    P = np.array([g.P_max for g in gens], float)
    s = P @ X - np.asarray(demand, float)
    return float(total + lambdas.lambda_1 * np.sum(s ** 2))


def exact_energy(x, gens, T, demand, lambdas, dt=1.0, initial_state=None) -> float:
    """QUBO energy (w = product) computed without building Q."""
    N = len(gens)
    init = list(initial_state) if initial_state is not None else [0] * N
    return exact_shifted_objective(x, gens, T, demand, lambdas, dt, init) - _offset(
        gens, demand, lambdas, init
    )


def full_vector(builder, x) -> np.ndarray:
    """Real variables plus w = x[t-1]*x[t], in the builder's index order."""
    xf = np.zeros(builder.n_vars)
    xf[: builder.n_vars_real] = x
    if builder.enable_mut_mdt:
        for i in range(builder.N):
            for t in range(1, builder.T):
                xf[builder.w_idx(i, t)] = x[builder.var_idx(i, t - 1)] * x[builder.var_idx(i, t)]
    return xf


# ---------------------------------------------------------------------------
# Tangent points for the squared imbalance
# ---------------------------------------------------------------------------

def _subset_sums(P) -> np.ndarray:
    sums = np.zeros(1)
    for p in P:
        sums = np.unique(np.round(np.concatenate([sums, sums + p]), 6))
    return sums


def _tangent_points(P, d_t, W, exact_ok, step) -> np.ndarray:
    total = float(np.sum(P))
    if exact_ok:
        a = _subset_sums(P) - d_t          # every achievable imbalance ...
        a = a[np.abs(a) <= W]              # ... inside the window
    else:
        a = np.arange(-W, W + step / 2.0, step)
    extra = [-W, W]
    reach = max(d_t, total - d_t)
    v = 2.0 * W
    while v < reach:                       # coarse tangents outside the window
        extra += [-v, v]
        v *= 2.0
    return np.unique(np.concatenate([a, np.array(extra)]))


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------

def solve_uc_milp(
    gens,
    T: int,
    demand,
    lambdas,
    dt: float = 1.0,
    initial_state: Optional[List[int]] = None,
    window_mw: float = 25.0,
    time_limit: float = 300.0,
    mip_rel_gap: float = 1e-9,
    max_exact_subset_n: int = 20,
    grid_step: float = 0.05,
) -> MilpResult:
    """
    window_mw : tangents are placed at every achievable imbalance with
        |s| <= window_mw (exact subset sums if N <= max_exact_subset_n, else a
        uniform grid of spacing grid_step), plus coarse tangents outside.
        Any schedule with some |s_t| > window costs more than
        lambda_1*window^2 in the penalty alone. If that exceeds the returned
        schedule's own shifted objective, the best schedule lies inside the
        window, where the tangent model is exact (subset-sum case only).
        That is what `window_certifies` reports.
    """
    N = len(gens)
    demand = np.asarray(demand, float)
    init = list(initial_state) if initial_state is not None else [0] * N
    P = np.array([g.P_max for g in gens], float)
    if window_mw <= 0:
        raise ValueError("window_mw must be > 0")
    for g in gens:
        if min(g.SU_cost, g.SD_cost, g.c_fuel) < 0:
            raise ValueError("indicator relaxation needs c_fuel, SU, SD >= 0")
    if min(lambdas.lambda_1, lambdas.lambda_3, lambdas.lambda_4) < 0:
        raise ValueError("lambdas must be >= 0")

    nx = N * T
    xi = lambda i, t: i * T + t
    ui = lambda i, t: nx + i * T + t
    di = lambda i, t: 2 * nx + i * T + t
    ri = lambda t: 3 * nx + t

    rows, cols, vals, lo, hi = [], [], [], [], []
    nrows = 0

    def add(entries, lb, ub=np.inf):
        nonlocal nrows
        for col, v in entries:
            rows.append(nrows)
            cols.append(col)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)
        nrows += 1

    n_extra = 0                    # v variables, appended after the r block
    v_cost = []

    def new_v(cost):
        nonlocal n_extra
        col = 3 * nx + T + n_extra
        n_extra += 1
        v_cost.append(cost)
        return col

    for i, g in enumerate(gens):
        mut, mdt = int(g.MUT), int(g.MDT)
        for t in range(T):
            if t == 0:
                add([(ui(i, 0), 1.0), (xi(i, 0), -1.0)], -init[i])      # u >= x0 - init
                add([(di(i, 0), 1.0), (xi(i, 0), 1.0)], init[i])        # d >= init - x0
            else:
                add([(ui(i, t), 1.0), (xi(i, t), -1.0), (xi(i, t - 1), 1.0)], 0.0)
                add([(di(i, t), 1.0), (xi(i, t), 1.0), (xi(i, t - 1), -1.0)], 0.0)
            for k in range(1, mut):
                if t + k >= T:
                    break
                v = new_v(lambdas.lambda_3)                              # v >= u - x_{t+k}
                add([(v, 1.0), (ui(i, t), -1.0), (xi(i, t + k), 1.0)], 0.0)
            for k in range(1, mdt):
                if t + k >= T:
                    break
                v = new_v(lambdas.lambda_4)                              # v >= d + x_{t+k} - 1
                add([(v, 1.0), (di(i, t), -1.0), (xi(i, t + k), -1.0)], -1.0)

    exact_ok = N <= max_exact_subset_n
    n_tangents = 0
    for t in range(T):
        for a in _tangent_points(P, float(demand[t]), window_mw, exact_ok, grid_step):
            # r_t >= 2a(sum_i P_i x_it - D_t) - a^2   (r_t stands for s_t^2)
            add(
                [(ri(t), 1.0)] + [(xi(i, t), -2.0 * a * P[i]) for i in range(N)],
                -2.0 * a * float(demand[t]) - a * a,
            )
            n_tangents += 1

    nvar = 3 * nx + T + n_extra
    c = np.zeros(nvar)
    for i, g in enumerate(gens):
        for t in range(T):
            c[xi(i, t)] = g.c_fuel * g.P_max * dt
            c[ui(i, t)] = g.SU_cost
            c[di(i, t)] = g.SD_cost
    c[3 * nx: 3 * nx + T] = lambdas.lambda_1
    c[3 * nx + T:] = np.array(v_cost)

    integrality = np.zeros(nvar)
    integrality[:nx] = 1
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[:nx] = 1.0

    A = coo_matrix((vals, (rows, cols)), shape=(nrows, nvar)).tocsr()
    res = milp(
        c,
        constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap},
    )

    offset = _offset(gens, demand, lambdas, init)
    dual = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    lower = (float(dual) - offset) if dual is not None else None
    if res.x is None:
        return MilpResult(res.status, res.message, None, None, lower, gap,
                          n_tangents, window_mw, None)

    x = np.rint(res.x[:nx]).astype(int)
    shifted = exact_shifted_objective(x, gens, T, demand, lambdas, dt, init)
    certifies = bool(exact_ok and lambdas.lambda_1 * window_mw ** 2 > shifted)
    return MilpResult(res.status, res.message, x, shifted - offset, lower, gap,
                      n_tangents, window_mw, certifies)


# ---------------------------------------------------------------------------
# Run on the 15-unit RTS-GMLC instance:  python -m fluxq.solvers.classical_lp
# ---------------------------------------------------------------------------

def _main() -> None:
    import os
    from pathlib import Path

    from ..data.grid_builder import make_demand_profile
    from ..data.loader import load_rts_gmlc_generators
    from ..qubo.builder import LambdaConfig, QUBOBuilder
    from ..qubo.validator import check_mut_mdt_feasibility, power_imbalance

    REF_CONST = -11_322_757_836.93      # best constant schedule, from your pasted output

    env = os.environ.get("RTS_GMLC_PATH")
    csv = Path(env) if env else (
        Path(__file__).resolve().parents[3] / "RTS-GMLC" / "RTS_Data" / "SourceData" / "gen.csv"
    )
    N, T = 15, 8
    gens = load_rts_gmlc_generators(csv)[:N]
    cap = sum(g.P_max for g in gens)
    demand = make_demand_profile(T=T, peak_demand=cap * 0.85, profile="flat")

    builder = QUBOBuilder(gens, T=T, enable_mut_mdt=True)
    lam1 = builder.suggest_lambda()
    lam_mm = builder.suggest_lambda_mut_mdt()
    lambdas = LambdaConfig(
        lambda_1=lam1 * 1.5, lambda_3=lam_mm, lambda_4=lam_mm,
        lambda_link=builder.suggest_lambda_link(lam_mm, lam_mm),
    )
    Q = builder.build(demand, lambdas)
    print(f"lambda_1={lambdas.lambda_1:,.5g} lambda_3={lambdas.lambda_3:,.5g} "
          f"lambda_link={lambdas.lambda_link:,.5g} n_vars={builder.n_vars} demand={demand[0]:.2f}")

    u_shift = REF_CONST + lambdas.lambda_1 * float(np.sum(demand ** 2))
    window = 1.25 * max(5.0, float(np.sqrt(u_shift / lambdas.lambda_1)))
    res = solve_uc_milp(gens, T, demand, lambdas, window_mw=window, time_limit=600.0)

    print(f"status={res.status} ({res.message})  tangents={res.n_tangents}  window={res.window_mw:.2f} MW")
    if res.x is None:
        print("no incumbent found; nothing to report")
        return
    xf = full_vector(builder, res.x)
    e_q = builder.evaluate(xf)
    print(f"MILP schedule energy (Q)   : {e_q:,.2f}   (module formula: {res.energy:,.2f}, "
          f"diff {e_q - res.energy:.3g})")
    print(f"MILP dual bound (as energy): {res.energy_lower_bound:,.2f}   mip_gap={res.mip_gap}")
    print(f"best constant (reference)  : {REF_CONST:,.2f}   MILP - constant = {e_q - REF_CONST:,.2f}")
    if res.energy_lower_bound is not None and res.energy_lower_bound > REF_CONST + 1.0:
        print("WARNING: dual bound above a known feasible energy; the model or setup is inconsistent")
    print(f"window_certifies           : {res.window_certifies}")
    X = res.x.reshape(N, T)
    print("units ON per step          :", X.sum(axis=0).tolist())
    imb = power_imbalance(xf, gens, demand, T)
    print(f"max |imbalance|            : {float(np.max(np.abs(imb))):.4f} MW")
    feas = check_mut_mdt_feasibility(xf, gens, T, initial_state=[0] * N)
    print(f"MUT/MDT feasible           : {feas['feasible']} (violations: {len(feas['violations'])})")
    print(f"costs                      : {builder.costs(xf)}")


if __name__ == "__main__":
    _main()