"""The five figures of spec section 6, drawn from results.csv alone.

This module never imports torch or transformers and never runs a model. Settings come from
the `plots` and `paths` sections of config.yaml, read as plain YAML so that importing
src.config (which needs torch) is not required.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.schema import KEY_COLUMNS, STABILITY_COLUMNS  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
SENTIMENT = "sentiment_a_mean"
# Figures 1 and 5 show both scorers; the others use scorer A.
SCORERS = {"A": SENTIMENT, "B": "sentiment_b_mean"}
REQUIRED_COLUMNS = (*KEY_COLUMNS, *SCORERS.values(), "perplexity", *STABILITY_COLUMNS)
NOISE_FLOOR_LABEL = "noise floor (std at strength 0)"
LANGS = ("en", "ar")
BASELINE_METHODS = ("steering", "random", "prompt_pos", "prompt_neg")


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def load_figure_files() -> dict[str, str]:
    return dict(_config()["plots"]["figure_files"])


def read_results(csv: Path | str, model: str | None = None) -> pd.DataFrame:
    """Read the CSV, check the columns the figures need, and keep a single model."""
    df = pd.read_csv(csv)
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"results CSV is missing required column '{column}'")
    models = sorted(df["model"].dropna().unique())
    if model is None:
        if len(models) != 1:
            raise ValueError(f"CSV holds models {models}; pass --model to choose one")
        model = models[0]
    return df[df["model"] == model].copy()


def _sentiment(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["vector_type"] == "sentiment"]


def noise_floor(df: pd.DataFrame, lang: str, column: str = SENTIMENT) -> float:
    """Std of the sentiment score at strength 0 over seeds, averaged over layers."""
    base = _sentiment(df)
    base = base[(base["lang"] == lang) & (base["strength"] == 0.0)]
    per_layer = base.groupby("layer")[column].std(ddof=1)
    return float(per_layer.mean())


def _mean_by(df: pd.DataFrame, by: list[str], column: str = SENTIMENT) -> pd.Series:
    return df.groupby(by)[column].mean()


def dose_response(df: pd.DataFrame) -> Figure:
    sent = _sentiment(df)
    fig, axes = plt.subplots(len(SCORERS), 2, figsize=(10, 7), sharey=True, sharex=True, squeeze=False)
    for row, (scorer, column) in zip(axes, SCORERS.items(), strict=True):
        for ax, lang in zip(row, LANGS, strict=True):
            sub = sent[sent["lang"] == lang]
            zero = float(sub[sub["strength"] == 0.0][column].mean())
            floor = noise_floor(df, lang, column)
            xs = sorted(sub["strength"].unique())
            ax.fill_between(xs, zero - floor, zero + floor, alpha=0.3, color="grey", label=NOISE_FLOOR_LABEL)
            for layer, g in sub.groupby("layer"):
                line = g.groupby("strength")[column].mean()
                ax.plot(line.index, line.values, marker="o", label=f"layer {int(layer)}")
            ax.set_title(f"{lang}, scorer {scorer}")
        row[0].set_ylabel(f"sentiment (scorer {scorer})")
    for ax in axes[-1]:
        ax.set_xlabel("strength (x mean residual norm)")
    axes[0][1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def transfer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per layer: shift at the largest strength, minus strength 0, in each language.

    The ratio is Arabic shift over English shift; it is NaN when the English shift is not
    larger than the English noise floor.
    """
    sent = _sentiment(df)
    top = float(sent["strength"].max())
    mean = _mean_by(sent, ["layer", "lang", "strength"])
    rows = []
    for layer in sorted(sent["layer"].dropna().unique()):
        shift = {lang: float(mean[(layer, lang, top)] - mean[(layer, lang, 0.0)]) for lang in LANGS}
        floor_en = noise_floor(df, "en")
        ratio = shift["ar"] / shift["en"] if abs(shift["en"]) > floor_en else float("nan")
        rows.append(
            {"layer": int(layer), "strength": top, "shift_en": shift["en"], "shift_ar": shift["ar"],
             "noise_en": floor_en, "ratio": ratio}  # fmt: skip
        )
    return pd.DataFrame(rows)


def transfer_ratio(df: pd.DataFrame) -> Figure:
    table = transfer_table(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    # Numeric x positions, so the axis works even when no layer has a defined ratio.
    x = list(range(len(table)))
    ax.bar(x, table["ratio"].fillna(0.0), color="tab:blue")
    for i, ratio in enumerate(table["ratio"]):
        if pd.isna(ratio):
            ax.text(i, 0, "undefined\n(English shift\nwithin noise)", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, table["layer"].astype(str))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"Arabic shift / English shift (strength {table['strength'].iloc[0]:g})")
    fig.tight_layout()
    return fig


def trade_off(df: pd.DataFrame) -> Figure:
    sent = _sentiment(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, LANGS, strict=True):
        sub = sent[sent["lang"] == lang]
        for layer, g in sub.groupby("layer"):
            m = g.groupby("strength")[[SENTIMENT, "perplexity"]].mean()
            ax.plot(m["perplexity"] - m.loc[0.0, "perplexity"], m[SENTIMENT] - m.loc[0.0, SENTIMENT],
                    marker="o", label=f"layer {int(layer)}")  # fmt: skip
        ax.set_title(lang)
        ax.set_xlabel("perplexity increase over strength 0")
    axes[0].set_ylabel("sentiment shift over strength 0")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def split_half(df: pd.DataFrame) -> Figure:
    sub = df[df["vector_type"] == "split_half"].sort_values("layer")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sub["layer"], sub["split_half_cos_mean"], yerr=sub["split_half_cos_std"], marker="o",
                capsize=3, label="mean +/- std")  # fmt: skip
    ax.plot(sub["layer"], sub["split_half_cos_min"], linestyle="none", marker="v", label="min")
    ax.set_xlabel("layer")
    ax.set_ylabel("split-half cosine of the vector")
    ax.legend()
    fig.tight_layout()
    return fig


def baseline_table(df: pd.DataFrame, layer: int | None) -> pd.DataFrame:
    """Sentiment shift over the strength-0 mean, per language, scorer and method.

    Steering and random use the largest strength. The layer is the one the random control
    was run at unless given. Prompting rows have no layer or strength.
    """
    if layer is None:
        rand_layers = df[df["vector_type"] == "random"]["layer"].dropna().unique()
        if len(rand_layers) != 1:
            raise ValueError(f"cannot pick the layer: random control found at {sorted(rand_layers)}; pass --layer")
        layer = int(rand_layers[0])
    top = float(_sentiment(df)["strength"].max())
    rows = []
    for lang in LANGS:
        sent = _sentiment(df)
        sent = sent[(sent["layer"] == layer) & (sent["lang"] == lang)]
        rand = df[(df["vector_type"] == "random") & (df["layer"] == layer) & (df["lang"] == lang)]
        for scorer, column in SCORERS.items():
            base = float(sent[sent["strength"] == 0.0][column].mean())
            floor = noise_floor(df, lang, column)
            values = {
                "steering": float(sent[sent["strength"] == top][column].mean()),
                "random": float(rand[rand["strength"] == top][column].mean()),
            }
            for method in ("prompt_pos", "prompt_neg"):
                values[method] = float(df[(df["vector_type"] == method) & (df["lang"] == lang)][column].mean())
            for method in BASELINE_METHODS:
                rows.append({"lang": lang, "scorer": scorer, "method": method, "shift": values[method] - base,
                             "noise": floor, "layer": layer})  # fmt: skip
    return pd.DataFrame(rows)


def steering_vs_baselines(df: pd.DataFrame, layer: int | None = None) -> Figure:
    table = baseline_table(df, layer)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, LANGS, strict=True):
        sub = table[table["lang"] == lang]
        x = list(range(len(BASELINE_METHODS)))
        width = 0.8 / len(SCORERS)
        for i, scorer in enumerate(SCORERS):
            s = sub[sub["scorer"] == scorer].set_index("method").loc[list(BASELINE_METHODS)]
            ax.bar([v + (i - (len(SCORERS) - 1) / 2) * width for v in x], s["shift"], width, yerr=s["noise"],
                   capsize=3, label=f"scorer {scorer}")  # fmt: skip
        ax.set_xticks(x, list(BASELINE_METHODS))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{lang}, layer {int(sub['layer'].iloc[0])}")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("sentiment shift over strength 0 (bars: noise floor)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def make_figures(df: pd.DataFrame, layer: int | None = None) -> dict[str, Figure]:
    return {
        "dose_response": dose_response(df),
        "transfer_ratio": transfer_ratio(df),
        "trade_off": trade_off(df),
        "split_half": split_half(df),
        "steering_vs_baselines": steering_vs_baselines(df, layer),
    }


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _config()
    parser = argparse.ArgumentParser(description="Draw the five figures from results.csv.")
    parser.add_argument("--csv", type=Path, default=Path(cfg["paths"]["results_csv"]))
    parser.add_argument("--out", type=Path, default=Path(cfg["paths"]["figures"]))
    parser.add_argument("--model", default=None, help="model id, needed if the CSV holds several")
    parser.add_argument("--layer", type=int, default=None, help="layer for figure 5 (default: the random control's)")
    args = parser.parse_args(argv)
    df = read_results(args.csv, args.model)
    figures = make_figures(df, args.layer)
    files = cfg["plots"]["figure_files"]
    args.out.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        fig.savefig(args.out / files[key], dpi=cfg["plots"]["dpi"])
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
