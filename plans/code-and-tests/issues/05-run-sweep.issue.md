# feat(code-and-tests): resumable sweep with controls [slice 05]

**Spec:** docs/steering_experiment_spec.md#53-sweep (also §5.4, §6, §8) · **Plan:** plans/code-and-tests/05-run-sweep.md
**Depends on:** #4, #5, #6
**Labels:** sub:code-and-tests

## Deliverable
`python -m src.run_sweep --config config.yaml [--grid smoke]` runs the whole sweep:
1. It extracts the vectors and writes the split-half rows.
2. It loops over layer × strength × seed × language with the sentiment vector.
3. It adds the random-direction and prompting-baseline conditions.

Each condition appends one row to `results/results.csv`, and every generation goes to `results/samples.jsonl`. Conditions already in the CSV are skipped.

## Mechanism
- This slice composes slices 02 to 04; the rest is plain Python. `pandas` reads the keys already in the CSV, each row is appended and flushed as soon as its condition finishes, and `json` writes the samples. The model and scorers are injectable, so the tests use the tiny model and the fakes.
- The best layer for the random direction is a config value, `random_direction_layer`, set by hand after the first pass. The sweep never picks it, so no tuning happens unseen.
- `--grid smoke` selects the reduced grid in `config.yaml` for phase 3.

## Acceptance criterion
- **Given** the tiny model, fake scorers and a grid of two layers, two strengths and two seeds,
- **When** the sweep runs, is interrupted after k conditions (simulated by a callback that raises), and then runs again,
- **Then**:
  - the final CSV has exactly one row per key `(model, layer, strength, seed, lang, vector_type)`, with no duplicates;
  - the second run generates only the missing conditions;
  - strength-0 rows exist for every seed and language (the noise floor);
  - `random`, `prompt_pos` and `prompt_neg` rows are present;
  - every measure column listed in `src/schema.py` is filled;
  - `samples.jsonl` has one line per prompt per generated condition.

## Files
```aeo-independence
slice: 05-run-sweep
creates: src/run_sweep.py
creates: tests/test_run_sweep.py
creates: docs/tdd-evidence/05-run-sweep.md
depends-on: 02-extract
depends-on: 03-steer
depends-on: 04-score
```

## Out of scope
The real smoke test (phase 3) and the T4 sweep (phase 4); the finer-layer pass; the Colab notebook.
