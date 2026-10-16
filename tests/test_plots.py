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
FINGERPRINT = "abc123def456"
OTHER_FINGERPRINT = "999888777666"
GIT_SHA = "0123abc"
# Shift over the strength-0 mean, by prompt placement: system and user differ on purpose.
PROMPT_SHIFTS = {
    "prompt_pos_system": 0.3,
    "prompt_neg_system": -0.3,
    "prompt_pos_user": 0.5,
    "prompt_neg_user": -0.5,
}
LAYERS = (4, 8)
STRENGTHS = (-1.0, 0.0, 1.0)
SEEDS = (0, 1, 2)
FIGURE_KEYS = ("dose_response", "transfer_ratio", "trade_off", "split_half", "steering_vs_baselines")


def _row(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in RESULT_COLUMNS}
    row.update(model=MODEL, fingerprint=FINGERPRINT, git_sha=GIT_SHA, **kw)
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
        for vt, shift in PROMPT_SHIFTS.items():
            for seed in SEEDS:
                rows.append(_row(seed=seed, lang=lang, vector_type=vt, **_measures(0.1 + shift, 12)))
    for layer in LAYERS:
        rows.append(_row(layer=layer, vector_type="split_half",
                         split_half_cos_mean=0.9, split_half_cos_std=0.02, split_half_cos_min=0.8))  # fmt: skip
    return rows


def write_csv(tmp_path: Path, drop: str | None = None, extra_fingerprints: tuple[str, ...] = ()) -> Path:
    rows = synthetic_rows()
    for fp in extra_fingerprints:
        rows += [{**r, "fingerprint": fp} for r in synthetic_rows()]
    df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
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
    assert labels.count(plots.NOISE_FLOOR_LABEL) == 4  # one band per language and scorer panel


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
    en = table[(table["lang"] == "en") & (table["scorer"] == "A")].set_index("method")["shift"]
    assert en["steering"] == pytest.approx(0.4)
    assert en["random"] == pytest.approx(0.02)
    assert en["prompt_pos_system"] == pytest.approx(0.3)


def test_transfer_ratio_figure_when_every_layer_is_undefined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    df = df[~((df["layer"] == 4) & (df["vector_type"] == "sentiment"))]  # only layer 8 left, English flat
    assert plots.transfer_table(df)["ratio"].isna().all()
    plots.transfer_ratio(df)  # no bars to draw; must still render the "undefined" labels


def test_dose_response_has_a_row_per_scorer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    fig = plots.dose_response(plots.read_results(write_csv(tmp_path)))
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert len(fig.axes) == 4
    assert any("scorer A" in s for s in labels) and any("scorer B" in s for s in labels)


def test_figure_five_reports_both_scorers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    df["sentiment_b_mean"] = df["sentiment_b_mean"] * 2  # scorer B sees twice the shift
    table = plots.baseline_table(df, layer=None)
    en = table[(table["lang"] == "en") & (table["method"] == "steering")].set_index("scorer")["shift"]
    assert en["A"] == pytest.approx(0.4)
    assert en["B"] == pytest.approx(0.8)


def test_figure_five_reports_all_four_prompting_types_per_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    table = plots.baseline_table(df, layer=None)
    for scorer in ("A", "B"):
        en = table[(table["lang"] == "en") & (table["scorer"] == scorer)].set_index("method")["shift"]
        for method, shift in PROMPT_SHIFTS.items():
            assert method in plots.BASELINE_METHODS
            assert en[method] == pytest.approx(shift)
    assert set(PROMPT_SHIFTS) <= set(table["method"])


def test_read_results_keeps_a_single_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    df = plots.read_results(write_csv(tmp_path))
    assert set(df["fingerprint"]) == {FINGERPRINT}
    assert df["fingerprint"].map(type).eq(str).all()


def test_read_results_reads_a_numeric_looking_fingerprint_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots = load_plots(monkeypatch)
    path = write_csv(tmp_path, extra_fingerprints=("000123456789",))
    df = plots.read_results(path, fingerprint="000123456789")
    assert set(df["fingerprint"]) == {"000123456789"}


def test_read_results_with_several_fingerprints_raises_naming_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plots = load_plots(monkeypatch)
    path = write_csv(tmp_path, extra_fingerprints=(OTHER_FINGERPRINT,))
    with pytest.raises(ValueError, match="fingerprint") as err:
        plots.read_results(path)
    assert FINGERPRINT in str(err.value) and OTHER_FINGERPRINT in str(err.value)


def test_fingerprint_argument_and_cli_option_choose_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plots = load_plots(monkeypatch)
    path = write_csv(tmp_path, extra_fingerprints=(OTHER_FINGERPRINT,))
    df = plots.read_results(path, fingerprint=OTHER_FINGERPRINT)
    assert set(df["fingerprint"]) == {OTHER_FINGERPRINT}
    out = tmp_path / "figs"
    assert plots.main(["--csv", str(path), "--out", str(out), "--fingerprint", FINGERPRINT]) == 0
    assert len(list(out.glob("*.png"))) == 5


def test_failing_figure_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No random rows (the first pass): figure 5 fails, figures 1 to 4 are still saved, exit is non-zero."""
    plots = load_plots(monkeypatch)
    df = pd.DataFrame(synthetic_rows(), columns=list(RESULT_COLUMNS))
    path = tmp_path / "results.csv"
    df[df["vector_type"] != "random"].to_csv(path, index=False)
    out = tmp_path / "figs"
    assert plots.main(["--csv", str(path), "--out", str(out)]) == 1
    assert len(list(out.glob("*.png"))) == 4
    assert "steering_vs_baselines" in capsys.readouterr().err
