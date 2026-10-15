"""Slice 04: scorers and side-effect measures. CPU only, no downloads."""

import contextlib
import math
from collections.abc import Iterator

import pytest

from src.config import load_config
from src.score import (
    arabic_script_share,
    bootstrap_ci,
    distinct_bigram_share,
    output_length,
    perplexity,
    sentiment_scores,
)

Row = list[dict[str, object]]

CARDIFF = ["negative", "neutral", "positive"]
CAMEL = ["positive", "negative", "neutral"]


def fake_classifier(order: list[str], winner: str):
    def classify(texts: list[str]) -> list[Row]:
        return [
            [{"label": name, "score": 1.0 if name == winner else 0.0} for name in order]
            for _ in texts
        ]

    return classify


@pytest.mark.parametrize("order", [CARDIFF, CAMEL])
def test_always_positive_gives_plus_one(order: list[str]) -> None:
    assert sentiment_scores(["a", "b"], fake_classifier(order, "positive")) == [1.0, 1.0]


@pytest.mark.parametrize("order", [CARDIFF, CAMEL])
def test_negative_gives_minus_one_and_neutral_zero(order: list[str]) -> None:
    assert sentiment_scores(["a"], fake_classifier(order, "negative")) == [-1.0]
    assert sentiment_scores(["a"], fake_classifier(order, "neutral")) == [0.0]


def test_uppercase_two_label_scorer_is_handled() -> None:
    clf = fake_classifier(["NEGATIVE", "POSITIVE"], "POSITIVE")
    assert sentiment_scores(["a"], clf) == [1.0]


def test_scores_stay_in_range_for_mixed_probabilities() -> None:
    def clf(texts: list[str]) -> list[Row]:
        return [
            [
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.3},
                {"label": "positive", "score": 0.5},
            ]
            for _ in texts
        ]

    (score,) = sentiment_scores(["x"], clf)
    assert -1.0 <= score <= 1.0
    assert score == pytest.approx(0.3)


def test_bootstrap_contains_mean_and_is_deterministic() -> None:
    cfg = load_config("config.yaml").bootstrap
    values = [0.1, 0.4, -0.2, 0.9, 0.3, -0.5, 0.0]
    a = bootstrap_ci(values, cfg.n_resamples, cfg.ci, cfg.seed)
    b = bootstrap_ci(values, cfg.n_resamples, cfg.ci, cfg.seed)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi
    assert mean == pytest.approx(sum(values) / len(values))


def test_bootstrap_differs_across_seeds() -> None:
    values = [0.1, 0.4, -0.2, 0.9, 0.3, -0.5, 0.0]
    assert bootstrap_ci(values, 200, 0.95, 0) != bootstrap_ci(values, 200, 0.95, 1)


def test_bootstrap_empty_does_not_raise() -> None:
    mean, lo, hi = bootstrap_ci([], 100, 0.95, 0)
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


def test_arabic_share() -> None:
    assert arabic_script_share("مرحبا بكم في العالم 123 !") == pytest.approx(1.0)
    assert arabic_script_share("hello world 123 !") == 0.0
    assert arabic_script_share("") == 0.0
    assert arabic_script_share("   12 .,! ") == 0.0
    assert 0.0 < arabic_script_share("مرحبا hello") < 1.0


def test_distinct_bigram_share() -> None:
    assert distinct_bigram_share("") == 0.0
    assert distinct_bigram_share("one") == 0.0
    assert distinct_bigram_share("a b c d") == 1.0
    assert distinct_bigram_share("a b a b a b") == pytest.approx(2 / 5)


def test_output_length_counts_whitespace_tokens() -> None:
    assert output_length("") == 0
    assert output_length("  a  b\tc\n") == 3


def test_perplexity_finite_and_above_one(tiny_model, tiny_tokenizer) -> None:
    texts = ["the food was very good .", "the hotel was terrible and bad ."]
    ppl = perplexity(tiny_model, tiny_tokenizer, texts)
    assert math.isfinite(ppl) and ppl > 1.0


def test_perplexity_accepts_steering_context(tiny_model, tiny_tokenizer) -> None:
    entered: list[bool] = []

    @contextlib.contextmanager
    def ctx() -> Iterator[None]:
        entered.append(True)
        yield

    base = perplexity(tiny_model, tiny_tokenizer, ["the food was good ."])
    with_ctx = perplexity(tiny_model, tiny_tokenizer, ["the food was good ."], steering=ctx())
    assert entered == [True]
    assert with_ctx == pytest.approx(base)
    assert all(p.grad is None for p in tiny_model.parameters())
