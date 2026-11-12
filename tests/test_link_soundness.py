"""
Soundness of lambda_link: the joint (x, w) minimum of E = z @ Q @ z, with w FREE,
must equal the minimum over product schedules (w = x[t-1]*x[t]).

The joint minimum is computed exactly with no product assumption, using the fact
that (if Q has no w-w couplings, which is asserted) E is linear in each w bit
for fixed x, so the best w bit is 1 exactly when its coefficient is negative.
That shortcut is itself validated against full enumeration of all x and all w
on a small instance.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fluxq.qubo.builder import LambdaConfig, QUBOBuilder

LAM_MM = 400.0


def _gen(name, c, P, su, sd, mut, mdt):
    return SimpleNamespace(name=name, c_fuel=c, P_max=P, SU_cost=su, SD_cost=sd, MUT=mut, MDT=mdt)


G3 = [_gen("A", 20.0, 50.0, 300.0, 50.0, 3, 2),
      _gen("B", 25.0, 30.0, 150.0, 20.0, 2, 2),
      _gen("C", 40.0, 20.0, 40.0, 10.0, 1, 1)]
G2 = [_gen("D", 30.0, 60.0, 500.0, 80.0, 4, 3),
      _gen("E", 35.0, 40.0, 200.0, 30.0, 3, 2)]

D6 = np.array([50.0, 80.0, 100.0, 100.0, 80.0, 50.0])
D6b = np.array([40.0, 70.0, 100.0, 70.0, 40.0, 60.0])
D4 = np.array([40.0, 90.0, 90.0, 40.0])

# (gens, T, demand, initial_state): T >= 6, MUT/MDT up to 4/3
CASES = [
    pytest.param(G3, 6, D6, [0, 0, 0], id="3u-T6-off"),
    pytest.param(G3, 6, D6, [1, 0, 1], id="3u-T6-mixed"),
    pytest.param(G2, 6, D6b, [0, 0], id="2u-mut4-T6-off"),
    pytest.param(G2, 6, D6b, [1, 1], id="2u-mut4-T6-on"),
]


def _instance(gens, T, demand, init, link=None):
    builder = QUBOBuilder(gens, T=T, initial_state=init, enable_mut_mdt=True)
    lam = LambdaConfig(lambda_1=5.0, lambda_3=LAM_MM, lambda_4=LAM_MM)
    lam.lambda_link = (builder.suggest_lambda_link(LAM_MM, LAM_MM)
                       if link is None else link)
    return builder, builder.build(demand, lam)


def _split(builder, Q, T):
    N = builder.N
    nreal = N * T
    widx = np.array([builder.w_idx(i, t) for i in range(N) for t in range(1, T)])
    assert builder.n_vars == nreal + len(widx)
    assert sorted(widx.tolist()) == list(range(nreal, builder.n_vars))
    assert np.allclose(np.tril(Q, -1), 0.0), "Q is not upper-triangular"
    Qww = Q[np.ix_(widx, widx)]
    assert np.allclose(Qww - np.diag(np.diag(Qww)), 0.0), "w-w coupling present"
    Qrr = Q[:nreal, :nreal]
    A = Q[:nreal, widx] + Q[widx, :nreal].T      # x-w couplings, nreal x nw
    d = np.diag(Q)[widx]                          # w diagonal (w^2 = w)
    return nreal, widx, Qrr, A, d


def _minima(builder, Q, T):
    """(joint minimum with w free, minimum over product w) over all x."""
    N = builder.N
    nreal, widx, Qrr, A, d = _split(builder, Q, T)
    masks = np.arange(2 ** nreal, dtype=np.uint32)
    X = ((masks[:, None] >> np.arange(nreal, dtype=np.uint32)) & 1).astype(float)
    base = np.einsum("bi,bi->b", X @ Qrr, X)
    C = d + X @ A                                 # coefficient of each w bit
    Xr = X.reshape(-1, N, T)
    Wp = (Xr[:, :, :-1] * Xr[:, :, 1:]).reshape(-1, N * (T - 1))  # i-major, t=1..T-1
    e_prod = base + (Wp * C).sum(axis=1)
    e_joint = base + np.minimum(C, 0.0).sum(axis=1)
    return float(e_joint.min()), float(e_prod.min())


@pytest.mark.parametrize("gens,T,demand,init", CASES)
def test_decomposition_matches_builder(gens, T, demand, init):
    """My x/w decomposition of E agrees with builder.evaluate on random (x, w)."""
    builder, Q = _instance(gens, T, demand, init)
    nreal, widx, Qrr, A, d = _split(builder, Q, T)
    rng = np.random.default_rng(7)
    for _ in range(100):
        xr = rng.integers(0, 2, size=nreal).astype(float)
        w = rng.integers(0, 2, size=len(widx)).astype(float)
        v = np.zeros(builder.n_vars)
        v[:nreal] = xr
        v[widx] = w
        e_dec = xr @ Qrr @ xr + w @ (d + xr @ A)
        assert np.isclose(builder.evaluate(v), e_dec, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("gens,T,demand,init", CASES)
def test_joint_minimum_equals_product_minimum(gens, T, demand, init):
    builder, Q = _instance(gens, T, demand, init)
    e_joint, e_prod = _minima(builder, Q, T)
    assert e_joint <= e_prod + 1e-6                # joint can never be above product
    assert abs(e_joint - e_prod) <= 1e-6 * max(1.0, abs(e_prod)), (e_joint, e_prod)


@pytest.mark.parametrize("link", [None, 2 * LAM_MM])
def test_shortcut_matches_full_enumeration(link):
    """Validate the linear-in-w shortcut against ALL x and ALL w (14 bits)."""
    T = 4
    builder, Q = _instance(G2, T, D4, [0, 1], link=link)
    n = builder.n_vars
    assert n <= 20
    masks = np.arange(2 ** n, dtype=np.uint32)
    Z = ((masks[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(float)
    e_full = float(np.einsum("bi,bi->b", Z @ Q, Z).min())
    e_joint, _ = _minima(builder, Q, T)
    assert abs(e_full - e_joint) <= 1e-6 * max(1.0, abs(e_full)), (e_full, e_joint)


@pytest.mark.parametrize("gens,T,demand,init", CASES)
def test_negative_control_small_link(gens, T, demand, init):
    """lambda_link = 2*lambda_3 was the original bug. Prints the gap; run with -s.
    Asserts only the always-true direction, so this never fails on a clean model."""
    builder, Q = _instance(gens, T, demand, init, link=2 * LAM_MM)
    e_joint, e_prod = _minima(builder, Q, T)
    print(f"\n[control link=2*lam] joint={e_joint:.4f} product={e_prod:.4f} "
          f"gap={e_prod - e_joint:.4f}")
    assert e_joint <= e_prod + 1e-6
    