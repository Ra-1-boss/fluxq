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
