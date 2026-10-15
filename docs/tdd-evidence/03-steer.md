# TDD evidence: slice 03, steer (issue #5)

## Red (15 Oct 2026)

`python -m pytest -q`, run with `tests/test_steer.py` written and no `src/steer.py`:

```
E   ModuleNotFoundError: No module named 'src.steer'
ERROR tests/test_steer.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.17s
```

Why it fails: `src/steer.py` doesn't exist yet, so `steering`, `generate` and `random_unit_vector` are not implemented. That is the behaviour this slice adds.

## A second red, on the first implementation

The first `src/steer.py` registered its hook with a plain `register_forward_hook`. One test failed:

```
FAILED tests/test_steer.py::test_hook_adds_scaled_vector_at_every_position
E       Mismatched elements: 160 / 160 (100.0%)
1 failed, 102 passed in 0.55s
```

Why it fails: transformers 5 builds `hidden_states` from its own forward hook on each decoder layer, registered before ours, so `hidden_states[L + 1]` showed the unsteered residual even though layers after L received the steered one (the logits tests passed). Registering with `prepend=True` makes the recorded residual the steered one.

## Green (15 Oct 2026)

```
103 passed in 0.49s
```

That is the 7 new tests in `tests/test_steer.py` plus the 96 existing tests, on CPU with the tiny random model and no downloads. `config.yaml` and `src/config.py` did not change: `generate` takes the existing `SamplingConfig`, and the random-direction seed is already `random_direction.seed`.
