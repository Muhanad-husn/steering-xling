"""Columns of results/results.csv and the key the sweep resumes on (spec section 6)."""

import math
from collections.abc import Mapping
from typing import Any

KEY_COLUMNS = ("model", "layer", "strength", "seed", "lang", "vector_type")

# vector_type values. split_half rows hold stability only: strength, seed and lang are empty
# (founder decision, 15 Oct 2026), so figure 4 still comes from results.csv alone.
VECTOR_TYPES = ("sentiment", "random", "prompt_pos", "prompt_neg", "split_half")

MEASURE_COLUMNS = (
    "sentiment_a_mean",
    "sentiment_a_lo",
    "sentiment_a_hi",
    "sentiment_b_mean",
    "sentiment_b_lo",
    "sentiment_b_hi",
    "perplexity",
    "arabic_share",
    "distinct_bigram_share",
    "length_mean",
    "n_samples",
)

STABILITY_COLUMNS = ("split_half_cos_mean", "split_half_cos_std", "split_half_cos_min")

RESULT_COLUMNS = KEY_COLUMNS + MEASURE_COLUMNS + STABILITY_COLUMNS

ConditionKey = tuple[str, int | None, float | None, int | None, str | None, str]


def _empty(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def condition_key(row: Mapping[str, Any]) -> ConditionKey:
    """Normalise a row from Python or from the CSV to one hashable key, empty cells as None."""

    def opt(name: str, cast: type) -> Any:
        value = row[name]
        if _empty(value):
            return None
        return cast(float(value)) if cast is int else cast(value)

    return (
        str(row["model"]),
        opt("layer", int),
        opt("strength", float),
        opt("seed", int),
        opt("lang", str),
        str(row["vector_type"]),
    )
