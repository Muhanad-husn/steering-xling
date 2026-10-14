# TDD evidence: slice 01, config skeleton (issue #3)

## Red (15 Oct 2026)

`python -m pytest -q tests/test_config.py`, run before any code in `src/`:

```
E   ModuleNotFoundError: No module named 'src.config'
ERROR tests/test_config.py
1 error in 0.20s
```

Why it fails: `src/config.py` and `src/schema.py` don't exist yet, so nothing for the config, the seeding or the schema is implemented. That is the behaviour this slice adds.
