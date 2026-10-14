# feat(code-and-tests): the five figures from results.csv alone [slice 06]

**Spec:** docs/steering_experiment_spec.md#6-outputs (also §8) · **Plan:** plans/code-and-tests/06-plots.md
**Depends on:** #3
**Labels:** sub:code-and-tests

## Deliverable
`python -m src.plots --csv results/results.csv --out results/figures` writes the five figures and reads nothing but the CSV:
1. dose-response;
2. transfer ratio;
3. trade-off;
4. split-half stability;
5. steering vs prompting vs random direction at the best layer.

## Mechanism
- Library: `matplotlib` (Agg backend) with `pandas`. Plain line and bar charts, no seaborn.
- Transfer ratio: per layer, the Arabic shift divided by the English shift, computed from the CSV. When the English shift is within the noise floor, the ratio is shown as undefined rather than as a huge number.

## Acceptance criterion
- **Given** a synthetic `results.csv` that the test builds from the columns in `src/schema.py`,
- **When** `plots.main` runs with `torch` and `transformers` made unimportable (blocked in `sys.modules`),
- **Then** five non-empty PNG files exist under the expected names. The dose-response plot draws the noise-floor band (the std at strength 0 over seeds). A CSV missing a required column fails with an error that names the column.

## Files
```aeo-independence
slice: 06-plots
creates: src/plots.py
creates: tests/test_plots.py
creates: docs/tdd-evidence/06-plots.md
depends-on: 01-config-skeleton
```

## Out of scope
REPORT.md; picking the 10 example generations; choosing the best layer.
