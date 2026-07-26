# FluxQ

**Quantum-native unit commitment for the renewable grid.**

[![Phase](https://img.shields.io/badge/phase-0%20complete-green)]()
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

I started this project because I wanted to actually understand QUBO, not just read about it. FluxQ is what came out of that.

It's an open-source framework that reformulates the **Unit Commitment Problem (UCP)** — deciding which power generators to switch on or off, and when, to meet electricity demand at the lowest possible cost — as a **QUBO (Quadratic Unconstrained Binary Optimization)** problem, so it can be solved on gate-based quantum computers (QAOA via Qiskit), quantum annealers (D-Wave), or classical solvers.

Every formula in here was worked out by hand before it became code, and the results are checked against either brute-force enumeration or real generator data pulled from RTS-GMLC — not numbers invented to make a demo look clean. The [research log](docs/research-log.md) documents the whole process, mistakes and corrections included, not just the parts that worked on the first try.

---

## The problem

Every grid operator faces a daily decision: which generators — coal, gas, wind, solar — do you switch **on or off**, and **when**, across a 24–48 hour window, to meet demand at minimum cost?

With 20–30% renewable penetration, this problem is exponentially harder. Existing MILP solvers (CPLEX, Gurobi) break down under uncertainty and real-time redispatch requirements. FluxQ investigates whether QUBO/quantum approaches close the gap.

## Why QUBO

The UCP is natively binary: every generator at every timestep is a 0/1 decision. QUBO is the native language of binary combinatorial optimization:

```
minimize  x^T Q x     where x ∈ {0,1}^n, Q is an n×n real matrix
```

Every constraint becomes a penalty term in Q. Every objective term becomes a diagonal or off-diagonal entry. Once in QUBO form, the problem can be solved by:

- **QAOA** on a gate-based quantum computer (via Qiskit)
- **Quantum Annealing** on a D-Wave system
- **Simulated Annealing** (classical baseline)
- **Gurobi/CPLEX** (classical benchmark)

---

## What makes this different

Most published QUBO formulations of the UCP:
1. Ignore minimum up/down time constraints (too hard to encode)
2. Use energy throughput as a proxy for renewable uncertainty (physically wrong)
3. Never benchmark against Gurobi on real grid data
4. Never analyse QUBO size scaling vs grid size

**FluxQ is working toward all four.** Phase 0 is verified against both a brute-force-checkable synthetic system and a real 5-generator slice of RTS-GMLC data — with hand-derived expected values confirmed by the code, not just assumed. That's the standard the rest of the project holds itself to.

---

## Repository structure

```
fluxq/
├── fluxq/
│   ├── qubo/
│   │   ├── builder.py        ← QUBOBuilder: the main entry point
│   │   ├── objective.py      ← Fuel, startup, shutdown cost terms
│   │   ├── constraints.py    ← Power balance, reserve, MUT/MDT (Phase 1)
│   │   └── validator.py      ← Energy evaluation, decoding, reporting
│   ├── data/
│   │   ├── grid_builder.py   ← Generator dataclass, toy/IEEE14/RTS-GMLC samples
│   │   └── loader.py         ← pandapower / MATPOWER loader
│   ├── solvers/               ← SA, QAOA, LP wrappers (Phase 1–2)
│   └── metrics/               ← Benchmarking utilities (Phase 2)
├── tests/                     ← 43 unit tests
├── demos/
│   ├── toy_brute_force.py             ← Phase 0 synthetic verification
│   └── rts_gmlc_5gen_brute_force.py   ← First real-data verification
├── notebooks/                 ← Jupyter walkthroughs (Phase 0–5)
├── datasets/                  ← Real dataset references (RTS-GMLC, etc.)
└── docs/
    ├── research-log.md        ← Every paper read, formula derived, bug found — in order
    └── glossary.md            ← Every term and dataset field explained, once
```

---

## Quick start

```bash
pip install -r requirements.txt
```

```python
from fluxq.data.grid_builder import make_toy_generators
from fluxq.qubo.builder import QUBOBuilder, LambdaConfig
import numpy as np

# 3-generator toy system
generators = make_toy_generators()
builder    = QUBOBuilder(generators, T=4)
demand     = np.array([200., 280., 350., 220.])   # MW per hour

lam = builder.suggest_lambda()
Q   = builder.build(demand, LambdaConfig(lambda_1=lam * 1.5))

print(f"QUBO matrix: {Q.shape}, state space: 2^{builder.n_vars}")
# → QUBO matrix: (12, 12), state space: 2^12
```

### Run the verification demos

```bash
python demos/toy_brute_force.py               # synthetic 3-generator system
python demos/rts_gmlc_5gen_brute_force.py      # real RTS-GMLC generators
```

Both enumerate every possible schedule and confirm the QUBO-optimal solution matches the true, hand-verifiable optimum.

---

## QUBO formulation

### Variables

```
x_{i,t} ∈ {0,1}    i = generator index, t = timestep index
```

For N generators and T timesteps: **n = N × T binary variables**. QUBO index: `i * T + t`.

### Objective (Phase 0 — implemented)

```
min Σ_{i,t} [ c_i · P_i · x_{i,t} · Δt          (fuel cost)
            + SU_i · max(0, x_{i,t} - x_{i,t-1}) (startup cost)
            + SD_i · max(0, x_{i,t-1} - x_{i,t}) (shutdown cost) ]
```

All three terms are QUBO-native. For binary variables:

```
max(0, x_t - x_{t-1}) = x_t - x_t · x_{t-1}    →  diagonal + off-diagonal entry
```

### Constraints (Phase 0 — power balance)

**Power balance** (equality):

```
λ₁ · Σ_t (Σ_i P_i · x_{i,t} - D_t)²
```

Expanding the square gives diagonal terms `λ₁ · P_i · (P_i − 2D_t)` and off-diagonal terms `2λ₁ · P_i · P_j`.

**Choosing λ₁:** must dominate the maximum possible objective savings. Use `builder.suggest_lambda()` for an automatic lower bound.

### Phase 1 constraints (planned)

- Minimum Up Time (MUT)
- Minimum Down Time (MDT)
- Ramp rate limits
- Spinning reserve

---

## Build roadmap

| Phase | Status | Content |
|---|---|---|
| **0** | ✅ Complete | QUBO builder, power balance, brute-force verification (synthetic + real RTS-GMLC data) |
| **1** | 🔄 Next | MUT/MDT encoding, Simulated Annealing, LP benchmark |
| **2** | ⬜ Planned | QAOA via Qiskit, small-system quantum benchmarks |
| **3** | ⬜ Planned | Renewable scenario extension (NREL wind/solar data) |
| **4** | ⬜ Planned | Scaling analysis, Lagrangian decomposition |
| **5** | ⬜ Planned | arXiv preprint, IEEE TPS submission |

---

## Mathematical note on QUBO energies

FluxQ uses the **upper-triangular convention**:

```
E(x) = x^T Q x = Σ_i Q[i,i] · x_i + Σ_{i<j} Q[i,j] · x_i · x_j
```

The power balance penalty **drops the constant term** `D_t²` (which does not affect the argmin). This means absolute QUBO energies are shifted relative to the actual penalty value — use `validator.actual_total_cost()` for true operational costs.

---

## Data sources

| Dataset | Content | Status |
|---|---|---|
| **RTS-GMLC** | Real generator technical parameters (NREL/GMLC test system) | ✅ Integrated — see `fluxq/data/grid_builder.py::make_rts_gmlc_sample()` |
| MATPOWER | IEEE 14/30/118-bus test cases | `pip install pandapower` |
| NREL Wind Toolkit | Hourly US wind output | https://www.nrel.gov/grid/wind-toolkit.html |
| NREL Solar Resource | Hourly solar irradiance | https://nsrdb.nrel.gov |
| ERCOT load data | Texas hourly load profiles | https://www.ercot.com/gridinfo/load/load_hist |

---

## Research log and glossary

Every paper read, formula derived by hand, bug found, and decision made — in chronological order, including the mistakes: **[docs/research-log.md](docs/research-log.md)**

Every term, abbreviation, and dataset field used across the project, explained once: **[docs/glossary.md](docs/glossary.md)**

---

## Citation

If you use FluxQ in your research, please cite:

```bibtex
@software{fluxq2026,
  title  = {FluxQ: A QUBO Framework for Unit Commitment in Renewable-Heavy Power Grids},
  year   = {2026},
  url    = {https://github.com/Ra-1-boss/fluxq}
}
```

---

## License

Apache 2.0. See `LICENSE`.
