"""Config loading, seeding, the results schema and the tiny-model fixture (slice 01, issue #3)."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.config import load_config, seed_everything
from src.schema import KEY_COLUMNS, RESULT_COLUMNS, VECTOR_TYPES, condition_key
from conftest import build_tiny_model, build_tiny_tokenizer

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def test_config_matches_spec() -> None:
    cfg = load_config(CONFIG)
    assert cfg.model.id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.model.dev_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.layers == (4, 8, 12, 16, 20, 24)
    assert cfg.strengths == (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
    assert len(cfg.seeds) == 3
    assert cfg.languages == ("en", "ar")
    assert cfg.sampling.temperature == 0.7
    assert cfg.sampling.top_p == 0.9
    assert cfg.sampling.max_new_tokens == 60
    assert cfg.scorers.a == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    assert cfg.scorers.b["en"] == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    assert cfg.scorers.b["ar"] == "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    assert cfg.stability.n_splits == 20
    assert cfg.paths.results_csv == Path("results/results.csv")


def test_config_is_frozen() -> None:
    cfg = load_config(CONFIG)
    with pytest.raises(AttributeError):
        cfg.layers = (1,)  # type: ignore[misc]


def test_smoke_grid_is_a_reduction_of_the_full_grid() -> None:
    cfg = load_config(CONFIG)
    assert 0.0 in cfg.smoke.strengths
    assert len(cfg.smoke.layers) < len(cfg.layers)
    assert cfg.smoke.n_prompts > 0


def _write_without(tmp_path: Path, dotted: str) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    *parents, leaf = dotted.split(".")
    node = raw
    for p in parents:
        node = node[p]
    del node[leaf]
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


@pytest.mark.parametrize("dotted", ["layers", "sampling.top_p", "scorers.b", "paths.results_csv"])
def test_missing_key_is_named(tmp_path: Path, dotted: str) -> None:
    with pytest.raises(ValueError, match=dotted.replace(".", r"\.")):
        load_config(_write_without(tmp_path, dotted))


def test_unknown_key_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["sampling"]["top_k"] = 50
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"sampling\.top_k"):
        load_config(out)


def test_seed_everything_makes_all_three_generators_repeat() -> None:
    def draw() -> tuple[float, float, float]:
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    seed_everything(7)
    first = draw()
    seed_everything(7)
    assert draw() == first


def test_schema_key_and_columns() -> None:
    assert KEY_COLUMNS == ("model", "fingerprint", "layer", "strength", "seed", "lang", "vector_type")
    assert RESULT_COLUMNS[: len(KEY_COLUMNS)] == KEY_COLUMNS
    assert len(set(RESULT_COLUMNS)) == len(RESULT_COLUMNS)
    assert set(VECTOR_TYPES) == {"sentiment", "random", "prompt_pos_system", "prompt_neg_system",
                                 "prompt_pos_user", "prompt_neg_user", "split_half"}  # fmt: skip
    assert "git_sha" in RESULT_COLUMNS and "git_sha" not in KEY_COLUMNS
    for col in ("sentiment_a_mean", "sentiment_b_mean", "perplexity", "arabic_share",
                "distinct_bigram_share", "length_mean", "split_half_cos_mean"):
        assert col in RESULT_COLUMNS


def test_condition_key_treats_csv_and_python_values_alike() -> None:
    from_python = condition_key({"model": "m", "fingerprint": "0a1b2c3d4e5f", "layer": 4, "strength": 0.5, "seed": 1,
                                 "lang": "en", "vector_type": "sentiment"})
    from_csv = condition_key({"model": "m", "fingerprint": "0a1b2c3d4e5f", "layer": "4", "strength": "0.5", "seed": "1",
                              "lang": "en", "vector_type": "sentiment"})
    assert from_python == from_csv
    split = condition_key({"model": "m", "fingerprint": "f", "layer": 4, "strength": float("nan"), "seed": None,
                           "lang": "", "vector_type": "split_half"})
    assert split == ("m", "f", 4, None, None, None, "split_half")


def test_tiny_model_builds_offline_on_cpu(no_network: None) -> None:
    tok = build_tiny_tokenizer()
    model = build_tiny_model(len(tok))
    assert model.config.num_hidden_layers <= 4
    assert model.config.hidden_size <= 64
    assert next(model.parameters()).device.type == "cpu"
    ids = tok.apply_chat_template([{"role": "user", "content": "describe the hotel"}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model(**ids)
    assert out.logits.shape[-1] == len(tok)
    assert torch.isfinite(out.logits).all()
