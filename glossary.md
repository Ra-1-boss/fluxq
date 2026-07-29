# FluxQ Glossary

A running reference for every term, abbreviation, and dataset field used across FluxQ -- code, research log, and documentation. New entries get added as they come up; nothing here is finalized or exhaustive.

---

## QUBO and optimization terms

**QUBO** (Quadratic Unconstrained Binary Optimization) -- an optimization problem of the form `minimize x^T Q x`, where `x` is a vector of binary (0/1) variables and `Q` is a matrix of real numbers encoding both the objective and the constraints. "Unconstrained" refers to the mathematical form, not the underlying problem -- real constraints are folded into `Q` as penalty terms.

**Ising model** -- a model from physics mathematically equivalent to QUBO, using variables in `{-1, +1}` instead of `{0, 1}`. Quantum annealers (like D-Wave) natively solve Ising-form problems; QUBO and Ising are interchangeable via a simple variable substitution.

**QAOA** (Quantum Approximate Optimization Algorithm) -- a hybrid classical-quantum algorithm that runs on gate-based quantum computers (like those accessed via Qiskit) to find approximate solutions to QUBO/Ising problems.

**Penalty weight (lambda, written λ)** -- a scalar multiplier on a constraint's penalty term. Must be large enough that violating the constraint is never cheaper than satisfying it, but not so large it drowns out the real objective. See `suggest_lambda()` in `builder.py` for FluxQ's automatic estimate.

**Transformation #1** -- the general technique (from Glover, Kochenberger & Du) for converting any equality constraint `Ax = b` into a QUBO penalty: add `P*(Ax - b)^2`, expand, simplify using `x^2 = x`, fold the result into `Q`. This is how FluxQ's power balance constraint is encoded.

**Transformation #2** -- the specific one-line penalty `P*x*y` for a mutual-exclusivity constraint `x + y <= 1`.

**Binary identity** -- the fact that `x^2 = x` for any binary variable `x` in `{0, 1}`. This is what allows linear cost terms to sit on the diagonal of `Q` and lets expanded penalty terms simplify cleanly.

---

## Unit Commitment Problem (UCP) terms

**Unit Commitment Problem (UCP)** -- the decision, for a set of power generators across a time horizon, of which generators to turn on or off at each timestep to meet demand at minimum cost, subject to technical constraints.

**MUT** (Minimum Up Time) -- the minimum number of consecutive timesteps a generator must remain ON after starting up, before it's allowed to shut down again. Prevents unrealistic rapid cycling.

**MDT** (Minimum Down Time) -- the minimum number of consecutive timesteps a generator must remain OFF after shutting down, before it's allowed to start again.

**Ramp rate** -- the maximum rate (MW per unit time) at which a generator's output can change. Relevant for generators that can operate at partial output, not just fully ON/OFF.

**Spinning reserve** -- extra generation capacity kept online (but not necessarily fully utilized) so the grid can respond quickly to unexpected demand spikes or generator failures.

**Power balance constraint** -- the requirement that total generation exactly equals demand at every timestep.

**Dispatchable generator** -- a generator whose output can be actively controlled/scheduled by the grid operator (e.g., coal, gas, oil). Contrast with **must-take** generators.

**Must-take generator** -- a generator whose output is determined by external factors, not operator choice (e.g., solar output depends on sunlight, wind on wind speed). These don't have a meaningful "commitment" decision in the way dispatchable generators do.

**Baseload** -- a generator (typically nuclear, sometimes large coal) designed to run continuously at a constant output, not cycled on and off on normal operational timescales.

**Startup cost / Shutdown cost** -- the cost incurred when a generator transitions OFF-to-ON or ON-to-OFF, respectively. Often includes both a fuel-based component (heat needed to bring equipment up to operating temperature) and a fixed non-fuel component (labor, wear).

**Cold / Warm / Hot start** -- the three tiers of startup cost depending on how long a generator has been offline. A "cold start" (offline a long time, fully cooled) requires far more energy to restart than a "hot start" (offline only briefly, still warm). FluxQ's Phase 0 model uses a single conservative `SU_cost` (based on Cold start) rather than modeling all three tiers.

**Heat rate** -- a measure of generator fuel efficiency: how much fuel energy (in Btu) is required to produce one unit of electrical energy (in kWh). Lower heat rate = more efficient. Units in RTS-GMLC: **Btu/kWh** (confirmed by cross-checking real values against known plant efficiencies -- e.g., a heat rate of 7222 Btu/kWh implies ~47% efficiency, correct for a modern gas combined-cycle plant).

---

## RTS-GMLC dataset fields (from `gen.csv`)

**RTS-GMLC** (Reliability Test System - Grid Modernization Laboratory Consortium) -- a real, publicly available synthetic power system test dataset built by NREL/GMLC specifically for unit commitment and grid studies. Includes generator technical parameters, network topology, and a full year of load/wind/solar timeseries data. Repository: `github.com/GridMod/RTS-GMLC`.

| Column | Meaning | Maps to FluxQ field |
|---|---|---|
| `GEN UID` | Unique generator identifier | `name` |
| `Fuel` | Fuel/technology type (Oil, Coal, NG, Nuclear, Hydro, Solar, Wind, Sync_Cond, Storage) | used to filter dispatchable generators |
| `PMax MW` | Maximum power output | `P_max` |
| `PMin MW` | Minimum stable generation level | `P_min` |
| `Min Up Time Hr` | Minimum up time | `MUT` |
| `Min Down Time Hr` | Minimum down time | `MDT` |
| `Fuel Price $/MMBTU` | Fuel cost per million Btu | used to derive `c_fuel` |
| `HR_avg_0` | Average heat rate (Btu/kWh) | used to derive `c_fuel` |
| `Start Heat Cold/Warm/Hot MBTU` | Fuel energy needed to start, by offline-duration tier | used to derive `SU_cost` (Cold tier used as the conservative default) |
| `Non Fuel Start Cost $` | Fixed non-fuel startup cost | added to `SU_cost` |
| `Non Fuel Shutdown Cost $` | Fixed shutdown cost | `SD_cost` |
| `Ramp Rate MW/Min` | Maximum ramp rate | `ramp_rate` (Phase 1) |

**Sync_Cond** (Synchronous Condenser) -- a piece of grid equipment that looks like a generator but produces no real power (`MW Inj` = 0); it only provides reactive power support for voltage stability. Excluded from FluxQ's UC model since it has no commitment decision to make.

---

## FluxQ project terms

**FluxQ** -- this project. A QUBO framework for the Unit Commitment Problem in renewable-heavy power grids.

**Phase 0** -- the foundational implementation: QUBO builder, objective (fuel/startup/shutdown cost), power balance constraint, verified against a small brute-force-checkable toy system. Complete as of 2026-07-08.

**Phase 1** -- next stage: MUT/MDT constraint encoding, Simulated Annealing and classical LP solvers as benchmarks.

**Q matrix** -- the assembled matrix representing a specific FluxQ problem instance -- objective costs plus constraint penalties, ready to be minimized by a solver.

**DegQuBO** -- a separate, related project (privately held, Volthos IP) applying similar QUBO/degradation-aware techniques to battery-diesel dispatch scheduling at the individual site level, rather than grid-scale unit commitment. Not part of the public FluxQ repo.

---

*Last updated: 2026-07-26 -- entries added while wiring in real RTS-GMLC data for the first time.*
