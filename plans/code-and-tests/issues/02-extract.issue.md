# feat(code-and-tests): steering-vector extraction and split-half stability [slice 02]

**Spec:** docs/steering_experiment_spec.md#51-extract-the-steering-vector · **Plan:** plans/code-and-tests/02-extract.md
**Depends on:** #3
**Labels:** sub:code-and-tests

## Deliverable
For every decoder layer L, `extract_vectors(model, tokenizer, pairs)` returns three things: the raw vector mean(positive) − mean(negative), its unit version, and the mean residual norm at L. Activations are taken on the residual stream after layer L and averaged over each sentence's tokens. `split_half_stability(...)` returns per-layer cosine similarities over 20 random half splits.

## Mechanism
- Library: `transformers` with `output_hidden_states=True`. `hidden_states[L+1]` is the residual stream after decoder layer L. Padding is masked out of the token average. Reading activations needs no hooks.
- Plain torch computes the means, norms and cosines. The split count comes from `config.yaml`.

## Acceptance criterion
- **Given** the tiny model and a handful of synthetic pairs,
- **When** `extract_vectors` runs,
- **Then** each layer's raw and unit vectors have shape `[d_model]`, each unit vector has norm 1 (atol 1e-5), and each mean norm is positive and finite. Swapping positive and negative negates the raw vector. With a fixed seed, `split_half_stability` returns `n_splits` cosines per layer, all in [-1, 1], and two calls with the same seed return identical values.

## Files
```aeo-independence
slice: 02-extract
creates: src/extract.py
creates: tests/test_extract.py
creates: docs/tdd-evidence/02-extract.md
depends-on: 01-config-skeleton
```

## Out of scope
Writing stability rows to `results.csv` (slice 05); plotting (slice 06); the kill-line decision (the founder's).
