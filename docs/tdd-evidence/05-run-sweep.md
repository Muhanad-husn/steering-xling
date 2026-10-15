# TDD evidence: slice 05, run_sweep (issue #8)

## Red (15 Oct 2026)

`python -m pytest -q`, run with `tests/test_run_sweep.py` written and no `src/run_sweep.py`:

```
    import src.run_sweep as run_sweep_mod
E   ModuleNotFoundError: No module named 'src.run_sweep'
ERROR tests/test_run_sweep.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 5.36s
```

Why it fails: `src/run_sweep.py` doesn't exist yet, so `run_sweep`, `Grid`, `SweepData` and `grid_for` are not implemented. That is the behaviour this slice adds.

## Green (15 Oct 2026)

`python -m pytest -q tests/test_run_sweep.py`:

```
..........                                                               [100%]
10 passed in 29.92s
```

Full suite, `python -m pytest -q`:

```
113 passed in 8.58s
```

The resume test interrupts the first run after 5 conditions, with a callback that raises, and then runs again. The first run makes exactly 5 x 2 generate calls and the second makes only the remaining 27 x 2. The final CSV has one row per key, and `samples.jsonl` has one line per prompt per condition.

## Random direction skips strength 0 (founder decision, 15 Oct 2026)

At strength 0 the random direction is the unsteered model, which the sentiment rows at strength 0 already cover. Red: `5 failed, 6 passed` (row and call counts, controls, samples, and the new `test_random_has_no_strength_zero_rows`). Green: `114 passed`.
