"""Scorers and side-effect measures for one condition's generations (spec 5.5)."""

import contextlib
import math
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

Classifier = Callable[[list[str]], list[list[dict[str, Any]]]]

POSITIVE = "positive"
NEGATIVE = "negative"

# Unicode blocks for Arabic script: Arabic, Supplement, Extended-A, Presentation Forms A and B.
ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def load_classifier(model_id: str) -> Classifier:
    """Build a real scorer lazily, so importing this module never touches the network."""
    from transformers import pipeline

    return pipeline("text-classification", model=model_id, top_k=None)  # type: ignore[return-value]


def _probs_by_name(row: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {str(d["label"]).lower(): float(d["score"]) for d in row}


def sentiment_scores(texts: Sequence[str], classifier: Classifier) -> list[float]:
    """P(positive) - P(negative) per text, labels matched by name, never by index."""
    if not texts:
        return []
    out: list[float] = []
    for row in classifier(list(texts)):
        probs = _probs_by_name(row)
        if POSITIVE not in probs or NEGATIVE not in probs:
            raise ValueError(f"scorer labels {sorted(probs)} lack '{POSITIVE}'/'{NEGATIVE}'")
        out.append(max(-1.0, min(1.0, probs[POSITIVE] - probs[NEGATIVE])))
    return out


def bootstrap_ci(
    values: Sequence[float], n_resamples: int, ci: float, seed: int
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap interval. Empty input gives NaNs."""
    if len(values) == 0:
        return (math.nan, math.nan, math.nan)
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return (mean, min(float(lo), mean), max(float(hi), mean))


def _is_arabic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ARABIC_RANGES)


def arabic_script_share(text: str) -> float:
    """Arabic-block characters over all letters. Whitespace, digits, punctuation ignored."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(_is_arabic(c) for c in letters) / len(letters)


def distinct_bigram_share(text: str) -> float:
    """Unique word bigrams over total word bigrams. Fewer than two words gives 0."""
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def output_length(text: str) -> int:
    """Length in whitespace-separated words."""
    return len(text.split())


def perplexity(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    steering: contextlib.AbstractContextManager[Any] | None = None,
) -> float:
    """exp(mean next-token loss over all reference tokens), under no_grad.

    `steering` is an optional context manager from the caller (e.g. a hook); this module
    does not import steer.py.
    """
    total_loss = 0.0
    total_tokens = 0
    ctx = steering if steering is not None else contextlib.nullcontext()
    device = next(model.parameters()).device
    with torch.no_grad(), ctx:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            n_predicted = ids.shape[1] - 1
            if n_predicted < 1:
                continue
            loss = model(input_ids=ids, labels=ids).loss
            total_loss += float(loss) * n_predicted
            total_tokens += n_predicted
    if total_tokens == 0:
        return math.nan
    return math.exp(total_loss / total_tokens)
