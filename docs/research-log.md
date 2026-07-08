# FluxQ Research Log

A running record of what's been read, built, and understood — in the order it happened. This is the primary log for FluxQ's development. Every entry is dated and stands on its own.

---

## 2026-07-04 — QUBO fundamentals

**Read:** Glover, Kochenberger & Du, ["A Tutorial on Formulating and Using QUBO Models"](https://arxiv.org/abs/1811.11538) (arXiv:1811.11538) — in full.

**Understood:**

- The QUBO form: minimize `y = xᵀQx` where `x` is a vector of binary variables and `Q` is a real matrix that encodes the entire problem. Diagonal entries `Q[i,i]` are standalone linear costs; off-diagonal entries `Q[i,j]` are interaction costs between decisions `i` and `j`.

- The binary identity `x² = x` (true for any `x ∈ {0,1}`) is the mechanical core of the whole framework. It's what lets linear cost terms fold onto the diagonal, and what simplifies expanded penalty terms down to something that fits cleanly into `Q`.

- Constraints are not solved separately — they're converted into quadratic penalty terms and added directly into the objective. Two transformations matter most for FluxQ:
  - **Transformation #1** (general equality `Ax = b`): add `P·(Ax - b)²`, expand, apply `x²=x`, collect coefficients into `Q`. This is how FluxQ's power balance constraint gets encoded.
  - **Transformation #2** (`x + y ≤ 1`): a one-line penalty `P·xy` — non-zero only when both are 1, the forbidden case. Single off-diagonal entry, no expansion needed.

- Penalty weight `P` has to sit in a "Goldilocks region" — large enough that no infeasible solution ever scores better than a feasible one, but not so large it drowns the real objective and makes the optimizer indifferent between good and bad feasible solutions. Rule of thumb from the paper: 75–150% of the expected objective value.

**Worked by hand:** A 2-generator, 1-timestep unit commitment toy problem — minimize fuel cost subject to an exact power balance constraint, encoded as a full 2×2 Q matrix, solved by brute-force enumeration of all 4 binary combinations to confirm the QUBO minimum matches the constrained optimum.

**Gap identified (this is where FluxQ's contribution starts):** the paper's toolkit has no treatment of constraints that couple a variable to itself across *time* — minimum up-time and minimum down-time constraints for generators. Every worked example in the paper is a single-shot constraint over variables at one point in time. FluxQ's unit commitment problem needs `x_{i,t}` to stay linked to `x_{i,t+1}, ..., x_{i,t+k}` once a generator starts up. That requires auxiliary startup/shutdown indicator variables not covered here — next reading target.

**Next:** survey papers on QAOA-based unit commitment to see how others have handled the MUT/MDT gap, before writing `fluxq/qubo/builder.py`.

---

## 2026-07-08 -- Phase 0 verified end-to-end: environment and repository debugging session

**Context for anyone reading this:** this entry is longer than usual because today wasn't about reading a new paper or writing new math -- it was about discovering that most of Phase 0's code, which we believed was already built and correct, had never actually made it onto disk, and then discovering the disk itself wasn't where GitHub thought it was either. What follows is the full story of how both problems were found and fixed, in order, so anyone picking this up can understand exactly what state the repo is in and why. Every formula and every line of code referenced below had already been verified by hand against test cases before today -- today was purely about getting that verified code to actually run, and to actually reach GitHub.

**Starting point:** `builder.py`, `objective.py`, `constraints.py`, `validator.py`, and `grid_builder.py` had all been reviewed and hand-checked in detail -- every cost formula, every constraint penalty, every test case walked through manually and confirmed correct. But "verified in review" and "actually present and working on disk" turned out to be two different things. And, as the day went on, "working on disk" and "known to git" turned out to be a third, separate thing again.

---

### Part 1 -- getting the code to actually run

**1. `demos/` folder didn't exist**
The README referenced `demos/toy_brute_force.py`, but the folder was never created in the original scaffold, and the demo script itself only ever existed as a file uploaded for review -- it had never been saved to the local machine at all.
*Fix:* created `demos/`, restored the file into it.

**2. `builder.py` was empty -- 0 bytes**
It had been created early on using a placeholder-generation step (`touch`), which reserves a filename but writes no content. The real content only ever existed in conversation, not on disk.
*Fix:* restored full file content. Also corrected a docstring inline: the documented formula for the power-balance penalty weight (`suggest_lambda()`) didn't mention that it includes worst-case startup cost -- the code was already right, the comment describing it wasn't.

**3. `grid_builder.py` didn't exist on disk at all -- not even as an empty file**
Different failure mode from #2: this file was simply never scaffolded in the first place, so it wasn't just empty, it was missing.
*Fix:* created it fresh with the `Generator` dataclass and the toy/IEEE-14 generator sets.

**4. `objective.py`, `constraints.py`, `validator.py` -- all empty, same as #2**
Same root cause: placeholder files that were never filled in.
*Fix:* restored all three. Applied the same docstring correction to `constraints.py` that was applied to `builder.py` in #2 -- both files had documented the same incomplete formula.

**5. `tests/` folder existed but was nested one level too deep**
It was sitting inside `fluxq\tests` (i.e., inside the Python package folder) instead of at the repo root `tests\`, which is where `pytest tests\` was actually looking.
*Fix:* moved it up one level.

**6. Even after fixing #5, `tests/` only contained `__init__.py`**
The two actual test files -- `test_qubo_builder.py` (27 tests) and `test_constraints.py` (11 tests) -- had the exact same problem as #2 and #4: they only existed in conversation, never on disk.
*Fix:* restored both files.

**7. `pytest` then failed with `ModuleNotFoundError: No module named 'fluxq.data.grid_builder'` -- despite the file genuinely existing with real content**
This was the trickiest one. Diagnosed by running direct Python import checks (`python -c "import fluxq; print(fluxq.__spec__)"`) rather than guessing from file listings. The output showed Python was treating `fluxq` as a **namespace package** rather than a normal one -- a fallback mode Python uses when it can't find a proper package marker.

Root cause: the **top-level** `fluxq/__init__.py` -- the marker for the outer package folder itself, separate from the ones inside `qubo/`, `data/`, `solvers/`, `metrics/` -- had never been created. Every subfolder had its own marker file; the folder containing all of them didn't.
*Fix:* created that one file. Confirmed the fix by checking that Python's import mechanism switched from `NamespaceLoader` to a normal `SourceFileLoader`.

**8. `fluxq.data` still failed to resolve `grid_builder.py` after #7 was fixed**
Traced to an incorrect relative path in an earlier move command that placed the file one folder too deep, creating a phantom nested `fluxq\fluxq\data\` structure that didn't correspond to anything Python actually imports.

**9. The first attempt to fix #8 made things worse -- created a triple-nested `fluxq\fluxq\fluxq\` folder**
This happened because relative paths were being typed from an assumption about the current folder, and the actual current folder had silently shifted after an earlier failed command. Relative paths compound errors invisibly when you're not 100% certain which folder you're standing in.
*Fix:* abandoned relative paths entirely. Got a full recursive listing of every `.py` file's complete absolute path in the whole repo in one shot, confirmed the true structure with certainty, then used complete absolute paths for every subsequent move -- no more guessing based on relative position.

**10. Cleanup -- dataset folders in the wrong place**
`matpower/` and `synthetic/` (meant for downloaded IEEE test case data, not Python code) had been dragged inside the Python package folder during the nesting confusion in #9.
*Fix:* moved them back out to a top-level `data/` folder, separate from the Python module.

**Checkpoint result:** `python -m pytest tests\ -v` -> **38 passed in 1.26s.** All formulas confirmed to actually run as reviewed. At this point the code was correct and working -- but, as Part 2 below found, it was working in the wrong folder entirely.

---

### Part 2 -- discovering the working code wasn't where GitHub thought it was

**11. `git push` failed with "no configured push destination," after creating a mysterious brand-new commit with no history**
This came as a surprise, since the repo had already been cloned and pushed to successfully days earlier. The commit message showed `(root-commit)` -- git's way of saying "this is the very first commit ever in this repository, nothing existed before it." That should never happen on a repo with existing history.

Root cause: the folder we'd been treating as "repo root" all day (`C:\Users\USER\Desktop\Projects\fluxq\`) was never actually a git repository at all -- it just *looked* like one because it contains a subfolder also named `fluxq\`, which genuinely is the real, GitHub-connected repo. When git commands ran with no `.git` present in that outer folder, git searched upward through parent folders looking for one, and found a completely unrelated, disconnected repository that had been accidentally created at the home-folder level (`C:\Users\USER\.git`) at some earlier point -- no remote configured, one orphan commit, nothing to do with this project. That's what silently absorbed the commit and then failed to push.

**12. Confirmed the true repo was one folder deeper than assumed all day**
`git remote -v`, run from inside `fluxq\fluxq\`, showed the real connection: `github.com/Ra-1-boss/fluxq.git`, with genuine prior history (the initial commit, the research log, the README fix). This explained a great deal of the day's earlier confusion in Part 1 -- every fix had been landing in the right *relative* position for Python's own imports to work (which is why `pytest` kept passing throughout), but in a folder git never knew existed. Two different tools, two different ideas of where "the project" lived, each working correctly in isolation from the other.

**13. Recovered by copying verified work into the real repo, not moving it blind**
Copied `tests/`, `demos/`, and the dataset folders (renamed `data/` -> `datasets/` on the way in, since the real repo's Python package already has its own, differently-purposed `data/` folder) from the outer container folder into the true repo at `fluxq\fluxq\`. Re-ran the full test suite from inside the real repo to confirm nothing was lost in the copy -- 38 passed again, this time from the correct location. Staged with `git add -A`, which correctly read most of the change as renames rather than delete-and-recreate pairs, confirming the content genuinely matched what git already expected. Committed and pushed.

---

### Final result

```
python -m pytest tests\ -v      (run from the real repo, fluxq\fluxq\)
===================== 38 passed in 3.63s =====================

git push
   e079327..c0070c7  main -> main
```

Confirmed live on GitHub at `github.com/Ra-1-boss/fluxq` -- `qubo/`, `data/`, `solvers/`, `metrics/`, `tests/`, `demos/`, `datasets/` all present.

---

### Lessons for the team

- **A file existing is not the same as a file having content.** Placeholder files created during initial scaffolding look identical to real files in a folder listing -- the only way to tell them apart is checking byte size, not just checking that the name is there.
- **Every Python package folder needs its own `__init__.py`, including the outermost one.** Missing it doesn't throw an obvious "package not found" error -- it produces a confusing, misleading error on a completely different, nested import instead. `python -c "import <package>; print(<package>.__spec__)"` is the fastest way to check -- `NamespaceLoader` in the output is the tell.
- **A folder can look like a repo root without being one.** Containing files, or containing a subfolder with the project's name, is not the same as being tracked by git. Confirm with `git remote -v` before assuming you're standing in the right place -- don't infer it from folder names alone.
- **If `git push` ever says "no configured push destination," or a commit shows `(root-commit)` on a repo that should already have history, stop immediately.** It means git silently found or created a different repository than the one intended, usually by searching upward through parent folders for a `.git` that wasn't where you thought. Verify `git remote -v` before pushing again.
- **When debugging folder or import issues, use absolute paths and full recursive listings, not incremental relative-path fixes.** Relative paths silently compound mistakes when the current working directory isn't certain.

---

### Current state

`fluxq\fluxq\` on the local machine is the true, GitHub-connected repository, and it now correctly contains the full Python package (`qubo/`, `data/`, `solvers/`, `metrics/`), `tests/`, `demos/`, and `datasets/`. All 38 tests pass from this location, and the latest commit is live on GitHub.

Two pieces of cleanup remain, tracked as follow-ups rather than done today: an orphaned, disconnected git repository sitting at the home-folder level (`C:\Users\USER\.git`, harmless but worth removing), and a leftover outer container folder (`C:\Users\USER\Desktop\Projects\fluxq\`, holding now-superseded copies of files that have already been correctly copied into the real repo).

To verify this yourself: clone the repo fresh, `pip install pytest numpy`, then run `python -m pytest tests\ -v` from the repo root. You should see `38 passed`.

**Next:** implement `add_mut_constraint` and `add_mdt_constraint` -- the Phase 1 research gap identified on 2026-07-04, still open.

---