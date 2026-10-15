# TDD evidence: slice 06, plots (issue #7)

## Red (15 Oct 2026)

`python -m pytest -q tests/test_plots.py`, run with `tests/test_plots.py` written and no `src/plots.py`:

```
E   ModuleNotFoundError: No module named 'src.plots'
FAILED tests/test_plots.py::test_main_writes_five_nonempty_pngs_without_torch
FAILED tests/test_plots.py::test_dose_response_draws_noise_floor_band
FAILED tests/test_plots.py::test_noise_floor_is_std_at_strength_zero_over_seeds
FAILED tests/test_plots.py::test_transfer_ratio_defined_and_undefined
FAILED tests/test_plots.py::test_missing_column_error_names_the_column
FAILED tests/test_plots.py::test_figure_five_uses_layer_of_random_control
6 failed in 0.57s
```

Why it fails: `src/plots.py` doesn't exist yet, so the five figures, the noise floor, the transfer table and the column check are not implemented. That is the behaviour this slice adds.

## Green (15 Oct 2026)

```
6 passed in 2.08s     (python -m pytest -q tests/test_plots.py)
109 passed in 5.80s   (python -m pytest -q)
```

The 6 new tests build a synthetic CSV in `tmp_path` from `src/schema.py` and run `plots.main` with `torch` and `transformers` set to `None` in `sys.modules`. `config.yaml` gained a `plots` section (`dpi`, `figure_files`) and `src/config.py` a matching `PlotsConfig`, because the loader refuses unknown keys. `src/plots.py` reads that section as plain YAML so it never imports `src.config`, which needs torch.

## A second red, found by feeding real sweep output (15 Oct 2026)

A CSV written by `src/run_sweep.py` (tiny model, fake scorers) crashed `python -m src.plots`. Every layer's English shift sat inside the noise floor, so the transfer-ratio chart had no bars, and placing the "undefined" label at a string x failed. The regression test `test_transfer_ratio_figure_when_every_layer_is_undefined` failed:

```
FAILED tests/test_plots.py::test_transfer_ratio_figure_when_every_layer_is_undefined
1 failed, 6 passed in 2.10s
```

Fix: numeric x positions with layer tick labels. Green: `110 passed in 2.24s`, and the sweep-written CSV now yields all five PNGs.
