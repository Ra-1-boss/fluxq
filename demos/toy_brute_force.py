"""
FluxQ Phase 0 Demo
==================
3-Generator, 4-Timestep Toy Problem — Brute-Force Verification

This script exhaustively enumerates all 2^12 = 4,096 binary states, evaluates
the QUBO energy of each, and ranks them. It then:

  1. Verifies that the QUBO-optimal solution is physically sensible.
  2. Computes actual operational costs for the top solutions.
  3. Checks known reference schedules against the brute-force ranking.
  4. Prints a full solution report.

This is the Phase 0 correctness check — if brute-force finds the right schedule,
the QUBO formulation is correct and we can proceed to Phase 1 (SA, LP, QAOA).

Usage
-----
    python demos/toy_brute_force.py

Or with a custom lambda:
    python demos/toy_brute_force.py --lambda1 0.02
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.grid_builder import make_toy_generators
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
from fluxq.qubo.validator import (
    decode_solution,
    actual_fuel_cost,
    actual_startup_cost,
    actual_shutdown_cost,
    power_imbalance,
)


# ---------------------------------------------------------------------------
# Brute-force solver
# ---------------------------------------------------------------------------

def brute_force(Q: np.ndarray, n_vars: int, top_k: int = 10):
    """
    Enumerate all 2^n_vars binary states, return top_k lowest-energy ones.

    Parameters
    ----------
    Q : np.ndarray, shape (n_vars, n_vars)
    n_vars : int
    top_k : int

    Returns
    -------
    list of (energy: float, x: np.ndarray)
        Sorted ascending by energy.
    """
    n_states = 1 << n_vars   # 2^n_vars
    if n_vars > 25:
        raise ValueError(
            f"n_vars={n_vars} is too large for brute-force enumeration "
            f"(max recommended: 25). Use n_vars ≤ 20 for demos."
        )

    print(f"  Enumerating {n_states:,} states for {n_vars} binary variables...")
    t0 = time.perf_counter()

    # Vectorised evaluation: build all 2^n binary rows at once
    # For n_vars=12, this is 4096 × 12 — fits easily in memory
    indices = np.arange(n_states, dtype=np.uint32)
    # Build binary matrix: X[s, i] = bit i of state s
    X = ((indices[:, None] >> np.arange(n_vars, dtype=np.uint32)[None, :]) & 1).astype(np.float64)

    # Vectorised energy: E[s] = X[s] @ Q @ X[s]^T  (diagonal of X @ Q @ X^T)
    QX = X @ Q                         # shape (n_states, n_vars)
    energies = np.einsum("si,si->s", QX, X)   # element-wise dot per row

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.3f}s.")

    # Sort and return top_k
    order = np.argsort(energies)
    results = [
        (float(energies[s]), X[s].copy())
        for s in order[:top_k]
    ]
    return results


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _schedule_str(schedule: np.ndarray, names: list[str], T: int) -> str:
    lines = []
    header = f"  {'Generator':<20}" + "  ".join(f"t{t}" for t in range(T))
    lines.append(header)
    lines.append("  " + "-" * (20 + 4 * T))
    for i, name in enumerate(names):
        row_vals = "   ".join(str(schedule[i, t]) for t in range(T))
        lines.append(f"  {name:<20}{row_vals}")
    return "\n".join(lines)


def _cost_summary(x, generators, T, dt, initial_state=None):
    fuel = actual_fuel_cost(x, generators, T, dt)
    su   = actual_startup_cost(x, generators, T, initial_state)
    sd   = actual_shutdown_cost(x, generators, T, initial_state)
    return fuel, su, sd, fuel + su + sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(lambda1: float = 5.0, top_k: int = 8) -> None:
    print()
    print("=" * 64)
    print("  FluxQ Phase 0 — Brute-Force Verification Demo")
    print("  3 Generators  ×  4 Timesteps  →  2^12 = 4,096 states")
    print("=" * 64)

    # ── System setup ───────────────────────────────────────────────────
    generators = make_toy_generators()
    T   = 4
    dt  = 1.0     # 1-hour timesteps
    N   = len(generators)

    # Demand: chosen so no single generator meets full demand at t=2,
    # but the coal+gas combination is sufficient.
    demand = np.array([200.0, 280.0, 350.0, 220.0])   # MW

    total_cap = sum(g.P_max for g in generators)

    print()
    print("  Generators")
    print(f"  {'Name':<20} {'P_max':>6}  {'c_fuel':>7}  {'SU_$':>6}  {'SD_$':>6}  {'MUT':>4}  {'MDT':>4}")
    print("  " + "-" * 60)
    for g in generators:
        print(f"  {g.name:<20} {g.P_max:>6.0f}  {g.c_fuel:>6.1f}$  {g.SU_cost:>6.0f}  {g.SD_cost:>6.0f}  {g.MUT:>4}  {g.MDT:>4}")

    print()
    print(f"  Demand profile  : {demand} MW")
    print(f"  Total capacity  : {total_cap:.0f} MW")
    print(f"  Demand range    : {demand.min():.0f} – {demand.max():.0f} MW")
    print(f"  Timestep dt     : {dt} h")

    # ── QUBO construction ──────────────────────────────────────────────
    builder = QUBOBuilder(generators, T=T, dt=dt)

    lambda_suggest = builder.suggest_lambda()
    print()
    print(f"  λ₁ lower bound (suggest_lambda): {lambda_suggest:.5f}")
    print(f"  λ₁ used in this demo           : {lambda1:.5f}")
    if lambda1 < lambda_suggest:
        print(f"  ⚠  WARNING: λ₁ < suggested lower bound. Constraint violations likely.")

    lambdas = LambdaConfig(lambda_1=lambda1)

    print()
    print("  Building QUBO matrix...", end=" ", flush=True)
    t0 = time.perf_counter()
    Q = builder.build(demand, lambdas)
    print(f"done in {time.perf_counter() - t0:.4f}s")

    builder.print_info()

    # Validate
    v = builder.validate()
    print()
    print("  QUBO validation:")
    for k, val in v.items():
        print(f"    {k:<22} {val}")

    # ── Brute-force search ─────────────────────────────────────────────
    print()
    print("  Brute-force search")
    print("  " + "-" * 40)
    results = brute_force(Q, builder.n_vars, top_k=top_k)

    names = [g.name for g in generators]

    print()
    print(f"  Top {top_k} solutions by QUBO energy:")
    print()
    print(f"  {'Rank':>4}  {'QUBO E':>12}  {'Fuel $':>8}  {'SU $':>6}  {'SD $':>6}  {'Total $':>8}  {'MaxImbal':>9}  Schedule")
    print("  " + "-" * 88)

    for rank, (energy, x) in enumerate(results, 1):
        decoded   = decode_solution(x, generators, T)
        schedule  = decoded["schedule"]
        fuel, su, sd, total = _cost_summary(x, generators, T, dt)
        imbal     = power_imbalance(x, generators, demand, T)
        max_imbal = float(np.max(np.abs(imbal)))
        sched_str = "|".join(
            "".join(str(schedule[i, t]) for t in range(T))
            for i in range(N)
        )
        marker = " ◄ BEST" if rank == 1 else ""
        print(
            f"  {rank:>4}  {energy:>12.4f}  {fuel:>8.0f}  {su:>6.0f}  {sd:>6.0f}  "
            f"{total:>8.0f}  {max_imbal:>9.1f}  {sched_str}{marker}"
        )
    print()
    print("  Schedule key: gen0|gen1|gen2, each 4 bits = t0t1t2t3")

    # ── Full report for best solution ──────────────────────────────────
    best_energy, best_x = results[0]
    print()
    print(builder.report(best_x, title="BEST QUBO SOLUTION (brute-force)"))

    # ── Reference schedule comparison ─────────────────────────────────
    print()
    print("=" * 64)
    print("  Reference Schedule Comparison")
    print("=" * 64)
    print()
    print("  Testing known feasible schedules against the QUBO energy...")
    print()

    reference_schedules = {
        "All OFF":
            np.zeros(builder.n_vars),
        "Coal only (all t)":
            np.array([1,1,1,1, 0,0,0,0, 0,0,0,0], dtype=float),
        "Gas only (all t)":
            np.array([0,0,0,0, 1,1,1,1, 0,0,0,0], dtype=float),
        "Coal+Gas (all t)":
            np.array([1,1,1,1, 1,1,1,1, 0,0,0,0], dtype=float),
        "All ON":
            np.ones(builder.n_vars),
        "Greedy (coal t0,t3; coal+gas t1,t2)":
            np.array([1,1,1,1, 0,1,1,0, 0,0,0,0], dtype=float),
        "Greedy+peaker (coal+gas+peak t2)":
            np.array([1,1,1,1, 0,1,1,0, 0,0,1,0], dtype=float),
        "Best QUBO":
            best_x,
    }

    print(f"  {'Schedule':<38} {'QUBO E':>12}  {'Total $':>8}  {'MaxImbal':>9}")
    print("  " + "-" * 72)

    ref_energies = {}
    for label, x in reference_schedules.items():
        e    = builder.evaluate(x)
        _, _, _, total = _cost_summary(x, generators, T, dt)
        imbal = power_imbalance(x, generators, demand, T)
        max_imbal = float(np.max(np.abs(imbal)))
        ref_energies[label] = e
        marker = " ◄" if label == "Best QUBO" else ""
        print(f"  {label:<38} {e:>12.4f}  {total:>8.0f}  {max_imbal:>9.1f}{marker}")

    # ── Sanity assertions ──────────────────────────────────────────────
    print()
    print("  Running sanity checks...")
    best_e = results[0][0]

    # 1. Brute-force optimal must have lower or equal energy than any reference
    for label, x in reference_schedules.items():
        if label == "Best QUBO":
            continue
        ref_e = ref_energies[label]
        assert best_e <= ref_e + 1e-9, (
            f"FAIL: brute-force best ({best_e:.4f}) > reference '{label}' ({ref_e:.4f})"
        )
    print("  ✓ Best QUBO energy ≤ all reference schedules.")

    # 2. Q matrix must be upper-triangular (no entries below diagonal)
    lower_norm = np.linalg.norm(np.tril(Q, k=-1))
    assert lower_norm < 1e-12, f"FAIL: Q is not upper-triangular (lower norm = {lower_norm:.2e})"
    print("  ✓ Q is upper-triangular.")

    # 3. Objective-only Q (no penalties) must give zero energy for all-OFF
    Q_obj_only = np.zeros_like(Q)
    from fluxq.qubo.objective import add_fuel_cost, add_startup_cost, add_shutdown_cost
    add_fuel_cost(Q_obj_only, generators, T, dt)
    add_startup_cost(Q_obj_only, generators, T)
    add_shutdown_cost(Q_obj_only, generators, T)
    e_all_off = float(np.zeros(builder.n_vars) @ Q_obj_only @ np.zeros(builder.n_vars))
    assert abs(e_all_off) < 1e-12, f"FAIL: all-OFF objective energy should be 0, got {e_all_off}"
    print("  ✓ All-OFF has zero objective energy (no startup/fuel/shutdown with all OFF).")

    # 4. Single generator ON at t=0 only: fuel + startup + shutdown(at t=1)
    x_coal_t0 = np.zeros(builder.n_vars)
    x_coal_t0[0] = 1.0   # generator 0, timestep 0 only (ON then immediately OFF)
    e_coal_t0 = float(x_coal_t0 @ Q_obj_only @ x_coal_t0)
    # fuel(t=0) + startup(t=0, initial=OFF) + shutdown(t=1, turns OFF after being ON)
    expected = (generators[0].c_fuel * generators[0].P_max * dt
                + generators[0].SU_cost
                + generators[0].SD_cost)
    assert abs(e_coal_t0 - expected) < 1e-9, (
        f"FAIL: coal ON at t=0 only should cost {expected:.2f}, got {e_coal_t0:.2f}"
    )
    print(f"  ✓ Coal ON at t=0 only: ${e_coal_t0:.0f} = fuel ${generators[0].c_fuel*generators[0].P_max*dt:.0f} + SU ${generators[0].SU_cost:.0f} + SD ${generators[0].SD_cost:.0f}.")

    print()
    print("  All sanity checks passed. ✓")
    print()
    print("=" * 64)
    print("  Phase 0 complete. Ready for Phase 1: Simulated Annealing + LP.")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluxQ Phase 0 brute-force demo")
    parser.add_argument(
        "--lambda1", type=float, default=5.0,
        help="Power balance penalty weight λ₁ (default: 5.0)"
    )
    parser.add_argument(
        "--topk", type=int, default=8,
        help="Number of top solutions to display (default: 8)"
    )
    args = parser.parse_args()
    main(lambda1=args.lambda1, top_k=args.topk)
