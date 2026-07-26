# FluxQ Research Log

A running record of what's been read, built, and understood -- in the order it happened. This is the primary log for FluxQ's development. Every entry is dated and stands on its own.

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
