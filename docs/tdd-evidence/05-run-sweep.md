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

## mean_norm leaves out the first token (founder decision, 15 Oct 2026)

The smoke run showed strength ±1 destroying the dev model, because the first token's attention-sink norm dominated `mean_norm` (see `logs/2026-10-15-smoke/`). The change is in `src/extract.py`, and the spec §5.1 is amended. Red: `test_mean_norm_excludes_each_sentence_first_token` failed (`1 failed, 5 passed`). Green: `115 passed`.

## Both prompt placements and a fingerprint in the key (founder decision, 15 Oct 2026)

Red: `tests/test_run_sweep.py` failed at collection with `ImportError: cannot import name 'fingerprint' from 'src.run_sweep'`. The new tests cover:
- the four `prompt_<tone>_<placement>` types;
- the instruction landing in the user turn for `user` and in the system prompt for `system`;
- one 12-hex fingerprint and a `git_sha` on every row;
- the fingerprint ignoring grid growth but changing with sampling settings and data;
- a rerun with changed settings producing new rows rather than reusing old ones.

The schema contract tests in `tests/test_config.py` were updated to the new key in the same change. Green: `120 passed`.
