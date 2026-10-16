# TDD evidence: figures saved one by one (issue #17, PR #18)

Found in the T4 smoke run on 16 Oct 2026: with no random rows, figure 5 raised and `main()` saved none of the five figures.

## Red (16 Oct 2026)

`python -m pytest -q tests/test_plots.py -k failing`, with the new test and the old `main()`:

```
src/plots.py:178: ValueError
FAILED tests/test_plots.py::test_failing_figure_does_not_stop_the_others - Va...
1 failed, 14 deselected in 2.27s
```

Why it fails: `make_figures` builds all five before any is saved, so the figure 5 `ValueError` escapes and nothing is written.

## Green (16 Oct 2026)

```
136 passed in 24.46s   (python -m pytest -q)
```

`main()` now builds and saves each figure on its own. A figure that raises `ValueError` is named on stderr and skipped, and `main()` returns 1. On the smoke run's `results.csv` it writes figures 1 to 4, prints `steering_vs_baselines: not drawn: cannot pick the layer: random control found at []; pass --layer`, and exits 1.
