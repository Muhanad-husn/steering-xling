# Slice 03: steering hook, random direction and seeded generation

**Spec:** docs/steering_experiment_spec.md#52-apply-it (also §5.3, §5.4) · **Issue:** [#5](https://github.com/Muhanad-husn/steering-xling/issues/5)
**Depends on:** 01-config-skeleton (#3)

## Goal
The context manager `steering(model, layer, unit_vector, mean_norm, alpha)` registers a forward hook on decoder layer L. The hook adds `alpha * mean_norm * unit_vector` to the layer output at every position, and it is removed when the context exits. `generate(...)` applies the chat template, with an optional system instruction for the prompting baseline. It seeds torch, numpy and `random`, then samples with the configured settings. `random_unit_vector(d_model, seed)` provides the random-direction control.

## Acceptance criterion
- **Given** the tiny model and a fixed input,
- **When** logits are computed inside `steering(..., alpha=0)` and again with no hook,
- **Then** the two sets of logits are identical. With alpha≠0 they differ. After the context exits, the hook is gone and the logits match the baseline again. Two `generate` calls with the same seed produce the same tokens. `random_unit_vector` has norm 1 and is deterministic in its seed.

## Mechanism
- Library: plain `torch.nn.Module.register_forward_hook` on `model.model.layers[L]`, handling the decoder layer's tuple output. No TransformerLens.
- `transformers` `apply_chat_template` and `model.generate(do_sample=True, temperature, top_p, max_new_tokens)`, with values from the config.

## Files
```aeo-independence
slice: 03-steer
creates: src/steer.py
creates: tests/test_steer.py
creates: docs/tdd-evidence/03-steer.md
depends-on: 01-config-skeleton
```

## Out of scope
Directional ablation (optional, phase 2); the sweep loop.
