# TDD evidence: figures 2 and 5 use the strength under the kill line's perplexity limit (17 Oct 2026)

Found while writing REPORT.md from the full 1.5B sweep. Figures 2 and 5 used the largest strength (+2), where every layer's output is broken (English perplexity 36 to 152,000 against a baseline of 6.0), so figure 2 showed transfer ratios from -4.6 to 6.0. The founder chose to draw, for each layer, the largest positive strength whose English perplexity is under 2x strength 0, which is the kill line's own limit in RULES.md. The founder also chose a symlog x-axis for figure 3.

The 2.0 lives in `src/plots.py` as `KILL_LINE_PERPLEXITY_RATIO`, not in `config.yaml`. Adding a key to `config.yaml` means changing `src/config.py`, which the fingerprint hashes. That would stop a later T4 run from reusing the 402 existing rows. This was the founder's decision.

## Red

`python -m pytest -q tests/test_plots.py -k "usable or limit or symlog"`, with the new tests and the old `plots.py`:

```
FAILED tests/test_plots.py::test_usable_strength_is_largest_under_the_perplexity_limit
FAILED tests/test_plots.py::test_transfer_ratio_ignores_strengths_past_the_perplexity_limit
FAILED tests/test_plots.py::test_figure_five_ignores_strengths_past_the_perplexity_limit
FAILED tests/test_plots.py::test_trade_off_uses_a_symlog_perplexity_axis - As...
4 failed, 15 deselected in 11.14s
```

## Green

```
140 passed in 77.43s   (python -m pytest -q)
```

On `results/results.csv`, figure 2 now uses +0.25 at L4 and L24, +1 at L8, L12 and L16, and +0.5 at L20. Figure 5 uses L12 +1, where steering B is +0.27 and random B is -0.32, matching the REPORT.md table.
