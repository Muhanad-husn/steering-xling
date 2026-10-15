"""Resumable sweep: vectors, split-half rows, steered conditions and controls (spec 5.3, 5.4, 6).

Run: `python -m src.run_sweep --config config.yaml [--grid smoke]`.

Each condition appends one row to results.csv and flushes it before the next starts, so a
crash loses at most the condition in progress. A rerun skips every key already in the CSV.
The samples of a condition are written just before its row; a crash between the two can
leave that condition's samples twice in samples.jsonl, keyed so they can be dropped.
"""

import argparse
import contextlib
import csv
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import Config, load_config
from src.extract import extract_vectors, split_half_stability
from src.schema import RESULT_COLUMNS, ConditionKey, condition_key
from src.score import (
    Classifier,
    arabic_script_share,
    bootstrap_ci,
    distinct_bigram_share,
    load_classifier,
    output_length,
    perplexity,
    sentiment_scores,
)
from src.steer import generate, random_unit_vector, steering

Record = Mapping[str, str]

# vector_type of each prompting-baseline row, and the instruction id it uses.
PROMPTING = {"prompt_pos": "pos", "prompt_neg": "neg"}


@dataclass(frozen=True)
class Grid:
    layers: tuple[int, ...]
    strengths: tuple[float, ...]
    seeds: tuple[int, ...]
    n_prompts: int | None  # per language; None means all prompts


@dataclass(frozen=True)
class SweepData:
    pairs: Sequence[Record]
    prompts: Sequence[Record]
    reference: Sequence[Record]
    instructions: Sequence[Record]


@dataclass(frozen=True)
class Condition:
    """One generated condition. Prompting rows have no layer, strength or vector."""

    vector_type: str
    layer: int | None
    strength: float | None
    seed: int
    lang: str
    unit: torch.Tensor | None = None
    mean_norm: float | None = None

    def key_fields(self, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "layer": self.layer,
            "strength": self.strength,
            "seed": self.seed,
            "lang": self.lang,
            "vector_type": self.vector_type,
        }


def grid_for(cfg: Config, smoke: bool) -> Grid:
    if smoke:
        s = cfg.smoke
        return Grid(layers=s.layers, strengths=s.strengths, seeds=s.seeds, n_prompts=s.n_prompts)
    return Grid(layers=cfg.layers, strengths=cfg.strengths, seeds=cfg.seeds, n_prompts=None)


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(cfg: Config) -> SweepData:
    p = cfg.paths
    return SweepData(
        pairs=read_jsonl(p.pairs),
        prompts=read_jsonl(p.prompts),
        reference=read_jsonl(p.reference),
        instructions=read_jsonl(p.instructions),
    )


def done_keys(results_csv: Path) -> set[ConditionKey]:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return set()
    df = pd.read_csv(results_csv, dtype={"model": str, "lang": str, "vector_type": str})
    return {condition_key(r) for r in df.to_dict("records")}


def append_row(results_csv: Path, row: Mapping[str, Any]) -> None:
    new = not results_csv.exists() or results_csv.stat().st_size == 0
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
        if new:
            writer.writeheader()
        writer.writerow({c: ("" if row.get(c) is None else row[c]) for c in RESULT_COLUMNS})
        f.flush()


def _by_lang(records: Sequence[Record], lang: str) -> list[Record]:
    return [r for r in records if r["lang"] == lang]


def _conditions(
    cfg: Config, grid: Grid, vectors_unit: torch.Tensor, mean_norm: torch.Tensor, d_model: int
) -> Iterator[Condition]:
    langs = cfg.languages
    for layer in grid.layers:
        for strength in grid.strengths:
            for seed in grid.seeds:
                for lang in langs:
                    yield Condition(
                        "sentiment", layer, strength, seed, lang, vectors_unit[layer], float(mean_norm[layer])
                    )
    rd = cfg.random_direction
    if rd.layer is None:
        print("random_direction.layer is not set in config.yaml: random-direction rows skipped")
    else:
        unit = random_unit_vector(d_model, rd.seed)
        # Strength 0 is the unsteered model, already run as the sentiment rows at strength 0.
        for strength in [s for s in grid.strengths if s != 0.0]:
            for seed in grid.seeds:
                for lang in langs:
                    yield Condition("random", rd.layer, strength, seed, lang, unit, float(mean_norm[rd.layer]))
    for vector_type in PROMPTING:
        for seed in grid.seeds:
            for lang in langs:
                yield Condition(vector_type, None, None, seed, lang)


def _steer_ctx(model: PreTrainedModel, c: Condition) -> contextlib.AbstractContextManager[Any]:
    if c.unit is None or c.layer is None or c.strength is None or c.mean_norm is None:
        return contextlib.nullcontext()
    return steering(model, c.layer, c.unit, c.mean_norm, c.strength)


def _system_prompt(data: SweepData, c: Condition) -> str | None:
    if c.vector_type not in PROMPTING:
        return None
    wanted = PROMPTING[c.vector_type]
    (text,) = [r["text"] for r in data.instructions if r["lang"] == c.lang and r["id"] == wanted]
    return text


def _sentiment_cols(prefix: str, scores: list[float], cfg: Config) -> dict[str, float]:
    b = cfg.bootstrap
    mean, lo, hi = bootstrap_ci(scores, b.n_resamples, b.ci, b.seed)
    return {f"{prefix}_mean": mean, f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def run_sweep(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    cfg: Config,
    grid: Grid,
    data: SweepData,
    scorer_a: Classifier,
    scorers_b: Mapping[str, Classifier],
    results_csv: Path,
    samples_path: Path,
    on_condition: Callable[[ConditionKey], None] | None = None,
) -> None:
    """Write split-half rows, then every missing condition, one flushed row at a time.

    `on_condition` is called after each generated condition's row is written.
    """
    done = done_keys(results_csv)
    batch = cfg.extraction.batch_size
    vectors = extract_vectors(model, tokenizer, data.pairs, batch_size=batch)

    def split_key(layer: int) -> ConditionKey:
        row = {"model": model_name, "layer": layer, "vector_type": "split_half"}
        return condition_key({**dict.fromkeys(RESULT_COLUMNS), **row})

    missing = [layer for layer in grid.layers if split_key(layer) not in done]
    if missing:
        st = cfg.stability
        cosines = split_half_stability(model, tokenizer, data.pairs, st.n_splits, st.seed, batch_size=batch)
    for layer in missing:
        cos = cosines[layer]
        row: dict[str, Any] = {"model": model_name, "layer": layer, "vector_type": "split_half"}
        row |= {
            "split_half_cos_mean": float(cos.mean()),
            "split_half_cos_std": float(cos.std(unbiased=False)),
            "split_half_cos_min": float(cos.min()),
        }
        append_row(results_csv, row)
        done.add(split_key(layer))

    d_model = int(vectors.unit.shape[-1])
    ppl_cache: dict[tuple[str, int | None, float | None, str], float] = {}
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    for c in _conditions(cfg, grid, vectors.unit, vectors.mean_norm, d_model):
        fields = c.key_fields(model_name)
        key = condition_key(fields)
        if key in done:
            continue
        prompts = _by_lang(data.prompts, c.lang)[: grid.n_prompts]
        system = _system_prompt(data, c)
        texts: list[str] = []
        with _steer_ctx(model, c), samples_path.open("a", encoding="utf-8") as f:
            for p in prompts:
                gen = generate(model, tokenizer, p["text"], cfg.sampling, c.seed, system=system)
                texts.append(gen.text)
                line = {**fields, "prompt_id": p["id"], "prompt": p["text"], "system": system, "text": gen.text}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()

        ppl_key = (c.vector_type if c.unit is not None else "none", c.layer, c.strength, c.lang)
        if ppl_key not in ppl_cache:
            ref = [r["text"] for r in _by_lang(data.reference, c.lang)]
            ppl_cache[ppl_key] = perplexity(model, tokenizer, ref, steering=_steer_ctx(model, c))

        row = {
            **fields,
            **_sentiment_cols("sentiment_a", sentiment_scores(texts, scorer_a), cfg),
            **_sentiment_cols("sentiment_b", sentiment_scores(texts, scorers_b[c.lang]), cfg),
            "perplexity": ppl_cache[ppl_key],
            "arabic_share": float(np.mean([arabic_script_share(t) for t in texts])) if texts else math.nan,
            "distinct_bigram_share": float(np.mean([distinct_bigram_share(t) for t in texts])) if texts else math.nan,
            "length_mean": float(np.mean([output_length(t) for t in texts])) if texts else math.nan,
            "n_samples": len(texts),
        }
        append_row(results_csv, row)
        done.add(key)
        if on_condition is not None:
            on_condition(key)


def main(argv: Sequence[str] | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grid", choices=["full", "smoke"], default="full")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    smoke = args.grid == "smoke"
    model_name = cfg.model.dev_id if smoke else cfg.model.id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.model.dtype) if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    run_sweep(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        cfg=cfg,
        grid=grid_for(cfg, smoke),
        data=load_data(cfg),
        scorer_a=load_classifier(cfg.scorers.a),
        scorers_b={lang: load_classifier(mid) for lang, mid in cfg.scorers.b.items()},
        results_csv=cfg.paths.results_csv,
        samples_path=cfg.paths.samples,
    )


if __name__ == "__main__":
    main()
