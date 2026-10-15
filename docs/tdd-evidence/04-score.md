# TDD evidence: slice 04, scorers and side-effect measures (issue #6)

## Red (15 Oct 2026)

`python -m pytest -q tests/test_score.py`, run before `src/score.py` existed:

```
E   ModuleNotFoundError: No module named 'src.score'
ERROR tests/test_score.py
1 error in 0.08s
```

Why it fails: `src/score.py` does not exist yet, so none of the sentiment mapping, bootstrap interval, perplexity, Arabic-script share, bigram share or length measures are implemented. That is the behaviour this slice adds.

## Green

(filled in after implementation)
