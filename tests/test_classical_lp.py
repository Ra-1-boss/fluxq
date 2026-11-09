"""
Cross-checks for fluxq.solvers.classical_lp on a tiny instance (3 units, T=5).

1. exact_energy() agrees with x @ Q @ x for random schedules (w = product).
2. The MILP optimum equals the minimum of x_full @ Q @ x_full over ALL 2^15
   product schedules, and its dual bound never exceeds that minimum.
Both are run with initial_state all-OFF and with some units ON, which
exercises the t=0 shutdown/MDT boundary and the SD offset.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from fluxq.qubo.builder import LambdaConfig, QUBOBuilder
from fluxq.solvers.classical_lp import exact_energy, full_vector, solve_uc_milp


def _gen(name, c, P, su, sd, mut, mdt):
    return SimpleNamespace(name=name, c_fuel=c, P_max=P, SU_cost=su, SD_cost=sd, MUT=mut, MDT=mdt)


GENS = [_gen("A", 20.0, 50.0, 300.0, 50.0, 3, 2),
        _gen("B", 25.0, 30.0, 150.0, 20.0, 2, 2),
        _gen("C", 40.0, 20.0, 40.0, 10.0, 1, 1)]
T = 5
DEMAND = np.array([50.0, 80.0, 100.0, 80.0, 50.0])


def _setup(init):
    builder = QUBOBuilder(GENS, T=T, initial_state=init, enable_mut_mdt=True)
    lam = LambdaConfig(lambda_1=5.0, lambda_3=400.0, lambda_4=400.0)
    lam.lambda_link = builder.suggest_lambda_link(400.0, 400.0)
    Q = builder.build(DEMAND, lam)
    return builder, lam, Q


@pytest.mark.parametrize("init", [[0, 0, 0], [1, 0, 1]])
def test_exact_energy_matches_qubo(init):
    builder, lam, _ = _setup(init)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = rng.integers(0, 2, size=len(GENS) * T)
        e_q = builder.evaluate(full_vector(builder, x))
        e_m = exact_energy(x, GENS, T, DEMAND, lam, initial_state=init)
        assert np.isclose(e_q, e_m, rtol=1e-9, atol=1e-6), (e_q, e_m)


@pytest.mark.parametrize("init", [[0, 0, 0], [1, 0, 1]])
def test_milp_matches_enumeration(init):
    builder, lam, Q = _setup(init)
    nbits = len(GENS) * T
    masks = np.arange(2 ** nbits, dtype=np.uint32)
    X = np.zeros((len(masks), builder.n_vars))
    X[:, :nbits] = (masks[:, None] >> np.arange(nbits)) & 1
    for i in range(builder.N):
        for t in range(1, T):
            X[:, builder.w_idx(i, t)] = X[:, i * T + t - 1] * X[:, i * T + t]
    E = np.einsum("bi,bi->b", X @ Q, X)
    e_min = float(E.min())

    res = solve_uc_milp(GENS, T, DEMAND, lam, initial_state=init, window_mw=1e3)
    assert res.status == 0
    assert abs(res.energy - e_min) < 1e-3, (res.energy, e_min)
    assert res.energy_lower_bound <= e_min + 1e-3
    assert abs(builder.evaluate(full_vector(builder, res.x)) - res.energy) < 1e-6