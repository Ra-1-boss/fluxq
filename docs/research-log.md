# FluxQ Research Log

A running record of what's been read, built, and understood -- in the order it happened. This is the primary log for FluxQ's development. Every entry is dated and stands on its own.

---

## 2026-07-29 -- Phase 1: MUT/MDT constraints implemented, the research gap since 2026-07-04 finally closed

**Context:** this is the actual research contribution FluxQ set out to make. Glover's QUBO tutorial (read 2026-07-04) has no worked example for constraints that couple a binary variable to itself across multiple future timesteps -- every example in it is a single-timestep constraint. Minimum Up Time and Minimum Down Time need exactly that: "if a generator starts at t, it must stay on through t+1, ..., t+MUT-1." This entry is the full derivation, the bug caught while deriving it, and the verification against real generator data.

**Why this is genuinely harder than Power Balance:** expanding the MUT violation condition -- `started_at_t * (1 - x_{t+k})` -- produces a term with **three** binary variables multiplied together: `x_t * x_{t-1} * x_{t+k}`. QUBO is strictly pairwise by definition; a cubic term cannot be written into a Q matrix directly. This is precisely the gap the July 4th entry flagged.

**The fix -- Rosenberg's cubic-to-quadratic reduction** (Rosenberg, 1975; mentioned only in passing, without a worked example, in Glover et al. section 7): introduce one auxiliary binary variable per (generator, timestep) pair, `w_{i,t}`, meant to equal the product `x_{i,t-1} * x_{i,t}`. Force it to that value with a penalty:

```
P_link(a, b, w) = a*b - 2*a*w - 2*b*w + 3*w
```

Verified by hand against all 8 combinations of `(a, b, w)` before writing any code: equals exactly 0 when `w = a*b`, strictly positive (1 or 3) otherwise. Substituting `w` for the product everywhere it appears collapses every cubic term back to degree <= 2. MUT and MDT both need the same product (it appears in both the startup and shutdown indicators), so they share one `w` block and one linking penalty rather than needing two.

**A real bug caught mid-derivation:** an early draft wrote the quadratic term `w_{i,t} * x_{i,t+k}` into the matrix as `Q[w, idx_tk]`. Since the auxiliary `w` block is appended after all real variables, `w`'s index is always larger than any real variable's index -- meaning this silently wrote into the *lower* triangle, breaking the "Q is strictly upper-triangular" structural invariant every other function in this codebase maintains. The bug would not have shown up as a wrong energy value (`evaluate()` sums over the full matrix regardless of triangle), but it would have broken `check_qubo_matrix()`'s off-diagonal range diagnostic, which only scans the upper triangle. Caught by re-deriving the index ordering explicitly for every term before writing the final code, not by testing after the fact.

**Hand-verified test cases, worked out before any code:**
- MUT=2, 1 generator, T=3, starting OFF: all 8 possible schedules checked against real-world feasibility by hand (e.g. "starts at t0, shuts off after 1 step" should violate; "starts at t0, stays 2 steps" should not). 3 infeasible, 5 feasible.
- MDT=2, 1 generator, T=3, starting ON: symmetric table, independently derived, same 3-infeasible/5-feasible split.

**Verification strategy -- three layers, strongest last:**
1. Direct `evaluate()` calls at each of the 8 hand-derived schedules (with `w` set to the true product) -- confirms the formula matches the hand-computed table exactly.
2. Full brute-force over all 32 states (8 x-patterns x 4 w-combinations) -- confirms the *minimum-energy* `w` for every schedule is exactly the true product, proving the linking penalty genuinely incentivizes the optimizer toward correct `w`, not just "gives the right answer when handed it."
3. Structural check against the real 5-generator RTS-GMLC sample -- confirms `Q` stays upper-triangular and `n_vars` grows exactly as `N*T + N*(T-1)` when applied to real data, not just the tiny synthetic case.

**Built:**
- `fluxq/qubo/constraints.py::add_transition_linking()` -- the Rosenberg substitution penalty.
- `fluxq/qubo/constraints.py::add_mut_constraint()` / `add_mdt_constraint()` -- replacing the `NotImplementedError` stubs from Phase 0.
- `fluxq/qubo/builder.py` -- `QUBOBuilder` gained a constructor-time `enable_mut_mdt` flag (deliberately constructor-time, not build()-time, since `n_vars` must be settled before `build()` is ever called -- e.g. before sizing a brute-force search), a `w_idx()` helper, and `suggest_lambda_link()` (a heuristic bound: a single wrong `w` can mask up to `(MUT-1)` MUT terms and `(MDT-1)` MDT terms simultaneously, so `lambda_link` must exceed the worst-case combined savings from cheating).
- `LambdaConfig` gained a `lambda_link` field.
- `tests/test_mut_mdt_constraints.py` -- 10 new tests covering all three verification layers above.
- `tests/test_constraints.py` -- removed the 2 now-obsolete stub tests (`test_mut_raises`, `test_mdt_raises`); `add_ramp_constraint` remains the one genuine Phase 1 stub left.

**Notable design win, not planned in advance:** `validator.py` needed zero changes. `decode_solution()` and every function built on it already slice `x[:N*T]` defensively -- a Phase 0 design choice made for a different reason (robustness) that turned out to make the auxiliary `w` block, appended after position `N*T`, completely invisible to every existing reporting function without touching them at all.

**Result:**
```
58 passed in 1.61s   (51 Phase 0/RTS-GMLC + 7 loader + 10 new MUT/MDT - 2 removed stubs)
```

Confirmed on the real machine after an unrelated environment hiccup (a second Python installation, 3.14, had taken priority on PATH ahead of the miniconda environment everything was actually installed in -- resolved by calling miniconda's `python.exe` directly; the PATH ordering itself deliberately left alone rather than untangling four separate Python installs mid-flow).

**Next:** wire MUT/MDT into a real solver run against the full 5-generator RTS-GMLC sample with real demand -- brute force stops being viable once auxiliary variables are added at any real scale (5 generators x 4 timesteps with MUT/MDT is already 35 variables, 2^35 states, far past brute-force range), so this is also the natural forcing function to finally start Simulated Annealing, still open since the very first FluxQ project brief.

---

## 2026-07-26 -- First real-data verification: 5 real RTS-GMLC generators, hand-reasoning confirmed by brute force

**Context:** after yesterday's full repository restart (new repo, clean single-layer folder structure, all verified Phase 0 code carried over unchanged), today's goal was to stop relying on synthetic toy data entirely and wire in a real, published grid dataset from the ground up -- starting with a small, deliberately hand-verifiable slice before scaling to anything larger.

**Dataset:** RTS-GMLC (Reliability Test System - Grid Modernization Laboratory Consortium), `github.com/GridMod/RTS-GMLC` -- a real, NREL/GMLC-built synthetic test system used across the power systems research community, containing full generator technical parameters plus a year of load/wind/solar timeseries.

**Explored:** `gen.csv` (158 total generators). Fuel type breakdown: Solar 57, NG 37, Hydro 20, Oil 19, Coal 16, Wind 4, Sync_Cond 3, Storage 1, Nuclear 1. Filtered to dispatchable thermal types only (Oil, Coal, NG) -- 72 generators survive the filter. Excluded Nuclear (can't cycle on UC timescales), Hydro (reservoir constraints not yet modeled), Solar/Wind (must-take, no real commitment decision), Sync_Cond (produces no real power, voltage support only), Storage (needs a different QUBO structure -- bidirectional flow, not on/off).

**Resolved two real unit-convention questions before writing any code:**
- Confirmed `HR_avg_0` (heat rate) is in **Btu/kWh**, not MMBTU/MWh, by cross-checking real values against known plant efficiency ranges (e.g. `107_CC_1` at 7222 Btu/kWh implies ~47% efficiency -- correct for a modern gas combined-cycle plant; the alternative unit convention would imply a physically impossible near-zero efficiency).
- Derived `c_fuel = HR_avg_0 * Fuel Price / 1000` and `SU_cost = Start Heat Cold MBTU * Fuel Price + Non Fuel Start Cost`, verified against real rows before generalizing.

**Selected 5 generators by hand** spanning the real merit order -- `223_STEAM_3` (350 MW coal baseload, cheapest, slowest to cycle: MUT=24h/MDT=48h) down to `101_CT_1` (20 MW oil peaker, most expensive, fastest to cycle: MUT=MDT=1h). Total capacity 856 MW. Chose this specific 5 deliberately: `n_vars = 5 x 4 timesteps = 20`, still within brute-force range (2^20 states), while being a genuine jump from synthetic data to real generator economics.

**Caught and corrected an arithmetic error mid-session:** initial by-hand SU_cost calculations for these 5 generators had small errors (largest: off by ~$4.21 on `223_STEAM_3`'s startup cost) from a decimal decomposition slip. Rechecked with a second independent decomposition before writing any code, and wrote the derivation formula directly into `make_rts_gmlc_sample()` rather than hardcoding hand-rounded numbers -- so the source values and the formula are both visible and independently checkable in the code itself, not hidden behind possibly-wrong manual arithmetic.

**Built:**
- `fluxq/data/grid_builder.py::make_rts_gmlc_sample()` -- the 5 real generators, computed from raw RTS-GMLC source fields transcribed directly from `gen.csv`.
- `tests/test_rts_gmlc_sample.py` -- 5 new tests verifying the derived values match hand computation within tolerance, plus a test protecting the merit-order/flexibility tradeoff property that motivated this particular selection.
- `demos/rts_gmlc_5gen_brute_force.py` -- brute-force verification against a hand-designed demand profile. Checked all 32 possible subsets of `{350, 76, 355, 55, 20}` MW by hand beforehand to confirm each of the 4 chosen demand values (350, 705, 856, 426 MW) has exactly one exact-match combination, making the brute-force result unambiguous to verify.

**Result:**
```
python -m pytest tests\ -v          -> 43 passed in 3.27s   (38 Phase 0 + 5 new)
python demos\rts_gmlc_5gen_brute_force.py
  -> Enumerated 1,048,576 states in 4.02s
  -> Best schedule found EXACTLY matches hand-reasoned prediction:
       223_STEAM_3 ON continuously (1,1,1,1) -- required at every timestep,
       pays its $36,749.81 startup cost exactly once
     101_STEAM_3, 107_CC_1, 315_CT_7, 101_CT_1 switch on/off exactly at the
     timesteps predicted by the by-hand subset-sum analysis
  -> Zero power imbalance at every timestep
  -> Total cost $144,040.40 (predicted by hand: ~$144,040.38 -- 2 cent
     difference is floating-point rounding only)
```

**Real finding, now empirically confirmed rather than theoretical:** startup cost is **56.7%** of total cost over this 4-hour window -- using real generator numbers, not an invented example. For a system built around large thermal units, scheduling that ignores startup-cost-aware constraints (exactly what MUT/MDT protects against) is missing over half the true cost structure. This is now a citable, defensible number for why Phase 1 matters, not an assumption.

**Later same day -- generalized the loader.** Wrote `fluxq/data/loader.py::load_rts_gmlc_generators()`, applying the same derivation formulas automatically across all 72 dispatchable generators instead of the 5 hand-transcribed ones. Rather than hand-verifying 72 rows of arithmetic (which would just reintroduce the exact transcription risk this work exists to remove), the validation strategy was to **cross-check the general loader against the 5 already-verified generators**: if it reproduces those exact values from the raw CSV, that's strong evidence of correctness across the rest.

`tests/test_rts_gmlc_loader.py::test_cross_validated_against_hand_verified_sample` confirms this -- the loader independently reproduces every hand-derived value (including the corrected `223_STEAM_3` startup cost) straight from the real file. Also added: a 72-count regression test, a Coal-only filter test (16, matching the earlier by-hand `Group-Object` count), and a clear `FileNotFoundError` for missing files rather than a confusing downstream crash.

**Known, deliberate gap:** the cross-validation test does not compare `P_min` -- the hand-verified sample never captured real `P_min` values, so it defaults to `0.0` there, while the general loader correctly reads real `P_min` from the CSV. Not a bug in either function, just an honest asymmetry worth fixing later if `P_min` becomes relevant (it isn't yet -- Phase 0's binary model doesn't use it).

```
python -m pytest tests\ -v   -> 50 passed in 1.32s   (43 + 7 new loader tests)

load_rts_gmlc_generators(..., verbose=True):
  158 total rows -> 72 kept (Coal, NG, Oil)
  Excluded: Solar 57, Hydro 20, Wind 4, Sync_Cond 3, Nuclear 1, Storage 1
  (matches the by-hand PowerShell count from earlier today, exactly)
```

**Next:** implement `add_mut_constraint`/`add_mdt_constraint` (still open from 2026-07-04). The full 72-generator real set, with its genuine MUT/MDT spread, is now loadable and ready to use once that work starts -- though brute-force verification won't scale past a handful of generators, so Phase 1 will also need a first real solver (Simulated Annealing) to search larger real systems.

---

## 2026-07-08 -- Phase 0 verified end-to-end: environment and repository debugging session

**Context for anyone reading this:** this entry is longer than usual because today wasn't about reading a new paper or writing new math -- it was about discovering that most of Phase 0's code, which we believed was already built and correct, had never actually made it onto disk, and then discovering the disk itself wasn't where GitHub thought it was either. What follows is the full story of how both problems were found and fixed, in order, so anyone picking this up can understand exactly what state the repo is in and why. Every formula and every line of code referenced below had already been verified by hand against test cases before today -- today was purely about getting that verified code to actually run, and to actually reach GitHub.

**Starting point:** `builder.py`, `objective.py`, `constraints.py`, `validator.py`, and `grid_builder.py` had all been reviewed and hand-checked in detail -- every cost formula, every constraint penalty, every test case walked through manually and confirmed correct. But "verified in review" and "actually present and working on disk" turned out to be two different things. And, as the day went on, "working on disk" and "known to git" turned out to be a third, separate thing again.

### Part 1 -- getting the code to actually run

**1. `demos/` folder didn't exist.** The README referenced `demos/toy_brute_force.py`, but the folder was never created in the original scaffold, and the demo script itself only ever existed as a file uploaded for review. *Fix:* created `demos/`, restored the file into it.

**2. `builder.py` was empty -- 0 bytes.** Created early on via a placeholder-generation step, which reserves a filename but writes no content. *Fix:* restored full file content, and corrected a docstring: the documented formula for the power-balance penalty weight (`suggest_lambda()`) didn't mention it includes worst-case startup cost -- the code was already right, the comment describing it wasn't.

**3. `grid_builder.py` didn't exist on disk at all.** Different failure mode: this file was simply never scaffolded in the first place. *Fix:* created it fresh.

**4. `objective.py`, `constraints.py`, `validator.py` -- all empty, same root cause as #2.** *Fix:* restored all three, and applied the same docstring correction to `constraints.py`.

**5. `tests/` folder existed but was nested one level too deep** -- inside the Python package folder instead of at the repo root. *Fix:* moved it up one level.

**6. `tests/` only contained `__init__.py`.** The actual test files (38 tests total) only existed in conversation, never on disk. *Fix:* restored both.

**7. `pytest` failed with `ModuleNotFoundError` for a file that genuinely existed.** Diagnosed via `python -c "import fluxq; print(fluxq.__spec__)"`, which showed Python treating `fluxq` as a **namespace package** -- a fallback mode triggered when a proper package marker is missing. Root cause: the top-level `fluxq/__init__.py` had never been created, even though every subfolder had its own. *Fix:* created that one file; confirmed via the import switching from `NamespaceLoader` to `SourceFileLoader`.

**8-9. Nested folder cascades from incorrect relative paths**, compounding because the current working directory wasn't certain at each step. *Fix:* abandoned relative paths entirely, used a full recursive absolute-path listing to confirm the true structure, then fixed with absolute paths only.

**10. Dataset folders (`matpower/`, `synthetic/`) had been dragged inside the Python package folder** during the nesting confusion. *Fix:* moved back to a top-level `data/` folder.

**Checkpoint result:** `python -m pytest tests\ -v` -> **38 passed in 1.26s.**

### Part 2 -- discovering the working code wasn't where GitHub thought it was

**11. `git push` failed with "no configured push destination,"** after creating a mysterious brand-new commit with no history (`root-commit`). Root cause: the folder being treated as "repo root" all day was never actually a git repository -- it just contained a subfolder with the same name that *was* the real repo. With no `.git` present where commands were running, git searched upward through parent folders and found a completely unrelated, disconnected repository that had been accidentally created at the home-folder level, with no remote configured.

**12. Confirmed the true repo was one folder deeper than assumed all day.** `git remote -v` showed the real GitHub connection with genuine prior history. This explained the day's earlier confusion -- every fix had landed in the right relative position for Python's imports (which is why `pytest` kept passing), but in a folder git never knew existed.

**13. Recovered by copying verified work into the real repo, not moving it blind.** Copied `tests/`, `demos/`, and dataset folders into the true repo location. Re-ran the full suite to confirm nothing was lost -- 38 passed again. Staged, committed, pushed successfully.

**Final result:**
```
python -m pytest tests\ -v      -> 38 passed in 3.63s
git push                        -> main -> main (confirmed live on GitHub)
```

**Lessons for the team:**
- A file existing is not the same as a file having content -- always check byte size, not just presence.
- Every Python package folder needs its own `__init__.py`, including the outermost one. A missing one produces a confusing error on an unrelated nested import, not an obvious "package not found."
- A folder can look like a repo root without being one. Confirm with `git remote -v` before assuming.
- If `git push` ever says "no configured push destination" or shows `(root-commit)` on a repo that should have history, stop and verify `git remote -v` before pushing again.
- Use absolute paths and full recursive listings when debugging folder/import issues, not incremental relative-path fixes.

**Next:** implement `add_mut_constraint` and `add_mdt_constraint` -- the Phase 1 research gap identified on 2026-07-04, still open.

---

## 2026-07-07 -- Phase 0 implemented: QUBO builder, objective, constraints, validator

**Built:**

- `fluxq/data/grid_builder.py` -- `Generator` dataclass (P_max, c_fuel, SU_cost, SD_cost, MUT, MDT, ramp_rate, initial_state) with full field validation, plus `make_toy_generators()` (3-gen system: coal baseload, gas combined-cycle, gas peaker) and `make_ieee14_generators()`.
- `fluxq/qubo/objective.py` -- fuel, startup, and shutdown cost terms encoded into Q. The core trick: `max(0, x_t - x_{t-1})` equals `x_t - x_t*x_{t-1}` exactly for binary variables, which converts "did this generator just switch on/off" into clean diagonal + off-diagonal QUBO entries.
- `fluxq/qubo/constraints.py` -- power balance and spinning reserve penalties via Transformation #1 (Glover et al.): `lambda*(sum P_i*x_i - target)^2`, expanded and folded into Q. `add_mut_constraint`, `add_mdt_constraint`, `add_ramp_constraint` are Phase 1 stubs that raise `NotImplementedError` by design.
- `fluxq/qubo/builder.py` -- `QUBOBuilder` orchestrates the above into one assembled Q matrix. Includes `suggest_lambda()`, an automatic lower-bound estimate for the power balance penalty weight, `info()` for QUBO size diagnostics, and full evaluate/decode/report methods.
- `fluxq/qubo/validator.py` -- energy evaluation, matrix structural checks, solution decoding, and actual cost calculation independent of QUBO encoding -- QUBO energy includes penalty terms and dropped constants, so it is never used as a stand-in for real operating cost.
- `demos/toy_brute_force.py` -- enumerates all 2^12 = 4,096 states of a 3-generator, 4-timestep toy problem, ranks by QUBO energy, verifies the winner matches the cheapest real-cost schedule.
- 38 unit tests, all passing -- each pins down an exact hand-computed QUBO energy for a known scenario.

**Verified by hand:** worked through the startup/shutdown cost encoding and the power balance penalty expansion manually for several test cases and confirmed every number the code produces against my own arithmetic.

**Bugs found and fixed:** `LambdaConfig` docstring and `add_power_balance` docstring both stated a lambda-1 lower-bound formula that omitted the startup-cost term actually used in `suggest_lambda()`. `QUBOBuilder.__init__` validated `initial_state` length but not that each value was 0 or 1 -- added the check. `demos/` folder didn't exist yet even though the README referenced it -- created it.

**Interesting finding:** a test (`test_cheaper_to_stay_on_than_cycle`) shows that without MUT/MDT constraints, the optimizer prefers cycling an expensive generator off and back on when fuel savings outweigh switching cost -- even though real generators can't do this safely. Direct evidence for why Phase 1's MUT/MDT encoding isn't optional.

**Next:** implement `add_mut_constraint` and `add_mdt_constraint` -- the research gap identified in the 2026-07-04 entry, where Glover's tutorial doesn't cover constraints that couple a variable to itself across multiple future timesteps. Needs auxiliary startup/shutdown indicator variables (`y_{i,t}`, `z_{i,t}`) per `docs/formulation.md`.

---

## 2026-07-04 -- QUBO fundamentals

**Read:** Glover, Kochenberger & Du, "A Tutorial on Formulating and Using QUBO Models" (arXiv:1811.11538) -- in full.

**Understood:**

- The QUBO form: minimize `y = x^T Q x` where `x` is a vector of binary variables and `Q` is a real matrix that encodes the entire problem. Diagonal entries are standalone linear costs; off-diagonal entries are interaction costs between decisions.
- The binary identity `x^2 = x` (true for any x in {0,1}) is the mechanical core of the whole framework. It's what lets linear cost terms fold onto the diagonal, and what simplifies expanded penalty terms down to something that fits cleanly into Q.
- Constraints are not solved separately -- they're converted into quadratic penalty terms and added directly into the objective. Two transformations matter most for FluxQ:
  - **Transformation #1** (general equality Ax = b): add `P*(Ax - b)^2`, expand, apply x^2=x, collect coefficients into Q. This is how FluxQ's power balance constraint gets encoded.
  - **Transformation #2** (x + y <= 1): a one-line penalty `P*xy` -- non-zero only when both are 1, the forbidden case.
- Penalty weight P has to sit in a "Goldilocks region" -- large enough that no infeasible solution ever scores better than a feasible one, but not so large it drowns the real objective. Rule of thumb: 75-150% of the expected objective value.

**Worked by hand:** A 2-generator, 1-timestep unit commitment toy problem -- minimize fuel cost subject to an exact power balance constraint, encoded as a full 2x2 Q matrix, solved by brute-force enumeration of all 4 binary combinations to confirm the QUBO minimum matches the constrained optimum.

**Gap identified (this is where FluxQ's contribution starts):** the paper's toolkit has no treatment of constraints that couple a variable to itself across *time* -- minimum up-time and minimum down-time constraints for generators. Every worked example in the paper is a single-shot constraint over variables at one point in time. FluxQ's unit commitment problem needs `x_{i,t}` to stay linked to `x_{i,t+1}, ..., x_{i,t+k}` once a generator starts up. That requires auxiliary startup/shutdown indicator variables not covered here.

**Next:** survey papers on QAOA-based unit commitment to see how others have handled the MUT/MDT gap, before writing `fluxq/qubo/builder.py`.

---
