"""Slice 06: the five figures come from results.csv alone, with no model libraries loaded."""

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.schema import MEASURE_COLUMNS, RESULT_COLUMNS, STABILITY_COLUMNS  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = (4, 8)
STRENGTHS = (-1.0, 0.0, 1.0)
SEEDS = (0, 1, 2)
FIGURE_KEYS = ("dose_response", "transfer_ratio", "trade_off", "split_half", "steering_vs_baselines")


def _row(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in RESULT_COLUMNS}
    row.update(model=MODEL, **kw)
    for m in MEASURE_COLUMNS:
        row.setdefault(m, None)
    return row


def _measures(sent: float, ppl: float) -> dict[str, Any]:
    return dict(
        sentiment_a_mean=sent, sentiment_a_lo=sent - 0.05, sentiment_a_hi=sent + 0.05,
        sentiment_b_mean=sent, sentiment_b_lo=sent - 0.05, sentiment_b_hi=sent + 0.05,
        perplexity=ppl, arabic_share=0.5, distinct_bigram_share=0.8, length_mean=30.0, n_samples=8,
    )  # fmt: skip


def synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for layer in LAYERS:
        for lang in ("en", "ar"):
            for s in STRENGTHS:
                for seed in SEEDS:
                    # Layer 4 moves in both languages; layer 8 barely moves English.
                    gain = {("en", 4): 0.4, ("ar", 4): 0.2, ("en", 8): 0.001, ("ar", 8): 0.3}[(lang, layer)]
                    jitter = 0.01 * (seed - 1)
                    sent = 0.1 + gain * s + jitter
                    rows.append(_row(layer=layer, strength=s, seed=seed, lang=lang,
                                     vector_type="sentiment", **_measures(sent, 10 + 3 * abs(s))))  # fmt: skip
    for lang in ("en", "ar"):
        for s in (0.0, 1.0):
            for seed in SEEDS:
                rows.append(_row(layer=4, strength=s, seed=seed, lang=lang,
                                 vector_type="random", **_measures(0.1 + 0.02 * s, 11)))  # fmt: skip
        for vt, shift in (("prompt_pos", 0.3), ("prompt_neg", -0.3)):
            for seed in SEEDS:
                rows.append(_row(seed=seed, lang=lang, vector_type=vt, **_measures(0.1 + shift, 12)))
    for layer in LAYERS:
        rows.append(_row(layer=layer, vector_type="split_half",
                         split_half_cos_mean=0.9, split_half_cos_std=0.02, split_half_cos_min=0.8))  # fmt: skip
    return rows


def write_csv(tmp_path: Path, drop: str | None = None) -> Path:
    df = pd.DataFrame(synthetic_rows(), columns=list(RESULT_COLUMNS))
    if drop:
        df = df.drop(columns=[drop])
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


def load_plots(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError
    monkeypatch.delitem(sys.modules, "src.plots", raising=False)
    return importlib.import_module("src.plots")


def test_main_writes_five_nonempty_pngs_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    out = tmp_path / "figs"
    assert plots.main(["--csv", str(write_csv(tmp_path)), "--out", str(out)]) == 0
    files = sorted(p.name for p in out.glob("*.png"))
    assert len(files) == 5
    assert all((out / f).stat().st_size > 0 for f in files)
    assert files == sorted(plots.load_figure_files().values())


def test_dose_response_draws_noise_floor_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    fig = plots.dose_response(df)
    labels = [c.get_label() for ax in fig.axes for c in ax.collections]
    assert labels.count(plots.NOISE_FLOOR_LABEL) == 2  # one band per language panel


def test_noise_floor_is_std_at_strength_zero_over_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    assert plots.noise_floor(df, "en") == pytest.approx(0.01)  # std of (-0.01, 0, 0.01), ddof=1


def test_transfer_ratio_defined_and_undefined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    table = plots.transfer_table(df).set_index("layer")
    assert table.loc[4, "ratio"] == pytest.approx(0.2 / 0.4)
    # English shift at layer 8 is 0.001, inside the 0.01 noise floor: undefined, not 300.
    assert math.isnan(table.loc[8, "ratio"])


def test_missing_column_error_names_the_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    out = tmp_path / "figs"
    with pytest.raises(ValueError, match="perplexity"):
        plots.main(["--csv", str(write_csv(tmp_path, drop="perplexity")), "--out", str(out)])
    assert STABILITY_COLUMNS[0] in RESULT_COLUMNS  # schema sanity


def test_figure_five_uses_layer_of_random_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    table = plots.baseline_table(df, layer=None)
    assert set(table["lang"]) == {"en", "ar"}
    en = table[table["lang"] == "en"].set_index("method")["shift"]
    assert en["steering"] == pytest.approx(0.4)
    assert en["random"] == pytest.approx(0.02)
    assert en["prompt_pos"] == pytest.approx(0.3)
