# CLAUDE.md: steering-xling

This repo is a dated, honest experiment. Can a sentiment steering vector built only from
English sentence pairs also steer `Qwen/Qwen2.5-1.5B-Instruct` when the prompt is in Arabic?
How large is the effect against a noise floor, a random direction and plain prompting, and
what does it cost in fluency? The audience reads the report for the measurement discipline,
not only for the effect.

## Where things are

| What | Where |
|---|---|
| The source of truth for the design | `docs/steering_experiment_spec.md` |
| Rules the experiment is held to, including the kill line | `RULES.md` |
| Phases, founder checklist, status table | `PLAN.md` |
| Per-slice specs and plans | `specs/`, `plans/` |
| Run records (every sweep, smoke test, failed run) | `logs/` |
| Evidence that a test failed before the code made it pass | `docs/tdd-evidence/` |
| Code | `src/` (`extract.py`, `steer.py`, `score.py`, `run_sweep.py`, `plots.py`) |
| Settings, all of them | `config.yaml` |
| Data written by the founder | `data/` (`pairs.jsonl`, `prompts.jsonl`, `reference.jsonl`) |
| Outputs | `results/results.csv`, `results/samples.jsonl`, `results/figures/` |

## Who may merge

**Only the founder decides a merge.** `main` is protected: every change goes through a pull
request, admins included, and no approving review is required because the founder works solo.

- An agent opens pull requests. It never merges on its own judgement.
- When the founder says "approved", "merge it" or anything equivalent, the agent does the
  rest without handing commands back: it squash-merges, deletes the branch locally and on
  the remote, removes any worktree that held it, syncs `main`, and reports the new SHA.
  Each of those runs as its own command, never chained.
- Nobody pushes directly to `main`.

## When code and spec disagree

The spec states intent. It does not outrank common sense.

1. If the code departs from the spec by accident, the code is wrong. Fix the code.
2. If following the spec would give an obviously wrong result, the spec is wrong. Examples:
   a scorer ID that no longer exists on Hugging Face, a layer index past the model's depth,
   a time budget the grid cannot meet. Do not follow the spec into the wrong answer, and do
   not quietly diverge either. Flag it in the pull request, propose the amendment, and once
   the founder agrees, edit `docs/steering_experiment_spec.md` in the same PR with a dated
   note saying what changed and why.
3. The spec itself requires scorer substitutions to be recorded. Record them in the spec's
   setup table and in REPORT.md.
4. Nobody changes `RULES.md` (the kill line especially) to make a result look better.
   Changing a rule is a founder decision, made before the result is seen.

## How work is done here

- **Tests first.** Unit tests run on CPU with a tiny randomly initialised model and no
  downloads. Required coverage:
  - the hook changes the output only when the strength is non-zero;
  - the vector has shape `[d_model]` per layer, and its unit version has norm 1;
  - sentiment scores stay in [-1, 1], and Arabic-script share and distinct-bigram share
    stay in [0, 1].
- **Test command:** `python -m pytest -q`, as recorded in `aeo-tests.json`. A pull request
  is green before it is proposed.
- **Python 3.11+, typed, minimal dependencies.** PyTorch, transformers, pandas, matplotlib
  and pyyaml, with plain forward hooks. No TransformerLens. Versions are pinned in
  `requirements.txt`.
- **No magic numbers in code.** Model, layers, strengths, seeds, sampling settings and
  paths all live in `config.yaml`.
- **Fixed seeds**, set for torch, numpy and python `random`, for each condition.
- **Steering maths.** The vector for layer L is mean(positive) minus mean(negative), taken
  on the residual stream after layer L and averaged over each sentence's tokens. The hook
  adds `alpha * mean_norm_L * unit_vector_L` at every position, so a strength means the
  same thing at every layer.
- **The sweep is resumable.** It skips any condition already in `results.csv`, keyed on
  model, layer, strength, seed, language and vector type.
- **Figures come from `results.csv` alone.** `plots.py` never reruns a model.
- **Development model:** `Qwen/Qwen2.5-0.5B-Instruct` on CPU. Real numbers come only from
  the 1.5B sweep on a T4.

## Reporting discipline

- Every number in REPORT.md has the noise floor (std at strength 0 over 3 seeds) beside it.
  An effect below the noise floor is not claimed.
- Best-layer results are always shown with the random-direction control and the prompting
  baseline.
- Arabic claims wait until the founder has read the Arabic outputs. A scorer's opinion is
  not a reading.
- The limits come in the first paragraph: small model, one behaviour, automatic scorers and
  their weaknesses on Arabic dialects, a single sweep run.
- The report is dated and calls itself an experiment run this week. A null result is
  reported as a result.

## Data

The founder writes the contrast pairs, the prompts and the Arabic (MSA) translations. An
agent may check format, counts and topic balance. It does not invent or rewrite the
founder's Arabic.

## Never

- Commit model weights, Hugging Face caches, tokens or `.env`.
- Tune the method after the kill line has fired, without the founder's decision.
- Report a figure that cannot be regenerated from `results.csv`.
