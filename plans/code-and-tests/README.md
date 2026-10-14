# Phase 2: Code and tests

**Spec:** docs/steering_experiment_spec.md (§3, §5, §6, §8) · **PLAN.md phase:** 2

Outcome: the five modules (`extract`, `steer`, `score`, `run_sweep`, `plots`) and `config.yaml` exist, each covered by CPU tests on a tiny randomly initialised model with no downloads. The pipeline is then ready for the phase 3 smoke test on Qwen2.5-0.5B.

| # | Slice | Depends on | Size | Parallel group | Issue |
|---|---|---|---|---|---|
| 01 | [Config, schema, tiny-model fixture](01-config-skeleton.md) | none | S | A | [#3](https://github.com/Muhanad-husn/steering-xling/issues/3) |
| 02 | [Vector extraction and split-half stability](02-extract.md) | 01 | S | B | [#4](https://github.com/Muhanad-husn/steering-xling/issues/4) |
| 03 | [Steering hook and seeded generation](03-steer.md) | 01 | S | B | [#5](https://github.com/Muhanad-husn/steering-xling/issues/5) |
| 04 | [Scorers and side-effect measures](04-score.md) | 01 | M | B | [#6](https://github.com/Muhanad-husn/steering-xling/issues/6) |
| 06 | [Five figures from results.csv](06-plots.md) | 01 | M | B | [#7](https://github.com/Muhanad-husn/steering-xling/issues/7) |
| 05 | [Resumable sweep with controls](05-run-sweep.md) | 02, 03, 04 | M | C | [#8](https://github.com/Muhanad-husn/steering-xling/issues/8) |

Groups were computed with `independence.mjs`: the full set exits 1 (dependencies only, no path collisions), and group B on its own exits 0.

## Decision: where the split-half stability lives (founder, 15 Oct 2026)

Stability goes into `results.csv` as rows with `vector_type=split_half`, empty strength, seed and language, and a `split_half_cos` column holding the mean, with std and min also recorded. Figure 4 therefore still comes from `results.csv` alone. A separate `stability.csv` was considered and rejected, because it would have needed an amendment to the "results.csv alone" rule.
