# TDD evidence: slice 02, extract (issue #4)

## Red (15 Oct 2026)

`python -m pytest -q`, run with `tests/test_extract.py` written and no `src/extract.py`:

```
E   ModuleNotFoundError: No module named 'src.extract'
ERROR tests/test_extract.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.10s
```

Why it fails: `src/extract.py` doesn't exist yet, so `extract_vectors` and `split_half_stability` are not implemented. That is the behaviour this slice adds.

## Green (15 Oct 2026)

GREEN_PLACEHOLDER
