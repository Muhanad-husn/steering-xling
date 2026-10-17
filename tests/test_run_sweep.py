"""Slice 05: resumable sweep with controls (spec 5.3, 5.4, 6)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.run_sweep as run_sweep_mod
from src.config import Config, SamplingConfig, load_config
from src.run_sweep import Grid, SweepData, fingerprint, grid_for, load_data, model_id_for, prompting_types, run_sweep
from src.schema import KEY_COLUMNS, MEASURE_COLUMNS, STABILITY_COLUMNS, condition_key

MODEL_NAME = "tiny"
LAYERS = (1, 2)
STRENGTHS = (0.0, 1.0)
SEEDS = (0, 1)
LANGS = ("en", "ar")
RANDOM_LAYER = 1
N_PROMPTS = 2
INTERRUPT_AFTER = 5

PAIRS = [
    {"positive": "the food was wonderful .", "negative": "the food was terrible ."},
    {"positive": "the hotel was great .", "negative": "the hotel was awful ."},
    {"positive": "i loved the film .", "negative": "i hated the film ."},
    {"positive": "work is good .", "negative": "work is bad ."},
]
PROMPTS = [
    {"id": "q01", "lang": "en", "text": "write two sentences about a train journey ."},
    {"id": "q02", "lang": "en", "text": "describe a kitchen table ."},
    {"id": "q01", "lang": "ar", "text": "اكتب جملتين عن رحلة بالقطار."},
    {"id": "q02", "lang": "ar", "text": "صف طاولة المطبخ."},
]
REFERENCE = [
    {"id": "r01", "lang": "en", "text": "the city is a city of trains ."},
    {"id": "r02", "lang": "ar", "text": "النيل نهر طويل."},
]
INSTRUCTIONS = [
    {"id": "pos", "lang": "en", "text": "respond in a very positive tone ."},
    {"id": "neg", "lang": "en", "text": "respond in a very negative tone ."},
    {"id": "pos", "lang": "ar", "text": "أجب بنبرة إيجابية جدًا."},
    {"id": "neg", "lang": "ar", "text": "أجب بنبرة سلبية جدًا."},
]

N_SENTIMENT = len(LAYERS) * len(STRENGTHS) * len(SEEDS) * len(LANGS)
# Random at strength 0 is the unsteered model, already in the sentiment rows: not rerun.
N_RANDOM = len([s for s in STRENGTHS if s != 0.0]) * len(SEEDS) * len(LANGS)
PLACEMENTS = ("system", "user")
N_PROMPTING = 2 * len(PLACEMENTS) * len(SEEDS) * len(LANGS)
N_GENERATED = N_SENTIMENT + N_RANDOM + N_PROMPTING


def fake_classifier(texts: list[str]) -> list[list[dict[str, Any]]]:
    """Deterministic scores from text length; label order varies to check matching by name."""
    out = []
    for t in texts:
        p = (len(t) % 7) / 7
        out.append([{"label": "Negative", "score": (1 - p) / 2}, {"label": "Positive", "score": p}])
    return out


class Interrupt(Exception):
    pass


@pytest.fixture
def cfg() -> Config:
    base = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return dataclasses.replace(
        base,
        languages=LANGS,
        sampling=SamplingConfig(temperature=0.7, top_p=0.9, max_new_tokens=4),
        stability=dataclasses.replace(base.stability, n_splits=3),
        random_direction=dataclasses.replace(base.random_direction, layer=RANDOM_LAYER),
        bootstrap=dataclasses.replace(base.bootstrap, n_resamples=20),
    )


@pytest.fixture
def grid() -> Grid:
    return Grid(layers=LAYERS, strengths=STRENGTHS, seeds=SEEDS, n_prompts=N_PROMPTS)


@pytest.fixture
def data() -> SweepData:
    return SweepData(pairs=PAIRS, prompts=PROMPTS, reference=REFERENCE, instructions=INSTRUCTIONS)


def _run(tmp_path: Path, model, tok, cfg, grid, data, on_condition=None) -> None:
    run_sweep(
        model=model,
        tokenizer=tok,
        model_name=MODEL_NAME,
        cfg=cfg,
        grid=grid,
        data=data,
        scorer_a=fake_classifier,
        scorers_b={lang: fake_classifier for lang in LANGS},
        results_csv=tmp_path / "results.csv",
        samples_path=tmp_path / "samples.jsonl",
        on_condition=on_condition,
    )


def _samples(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def resumed(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data, monkeypatch, no_network):
    """Run, interrupt after INTERRUPT_AFTER conditions, run again. Returns generate-call counts."""
    calls = {"n": 0}
    real_generate = run_sweep_mod.generate

    def counting_generate(*args: Any, **kwargs: Any):
        calls["n"] += 1
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(run_sweep_mod, "generate", counting_generate)
    done = {"n": 0}

    def interrupt(_key: Any) -> None:
        done["n"] += 1
        if done["n"] == INTERRUPT_AFTER:
            raise Interrupt

    with pytest.raises(Interrupt):
        _run(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data, on_condition=interrupt)
    first = calls["n"]
    _run(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data)
    return {"first": first, "second": calls["n"] - first}


def _generated(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["vector_type"] != "split_half"]


def test_one_row_per_key_after_resume(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    keys = [condition_key(r) for r in df.to_dict("records")]
    assert len(keys) == len(set(keys))
    assert len(_generated(df)) == N_GENERATED
    assert (df["vector_type"] == "split_half").sum() == len(LAYERS)


def test_second_run_generates_only_missing_conditions(resumed) -> None:
    per_condition = N_PROMPTS
    assert resumed["first"] == INTERRUPT_AFTER * per_condition
    assert resumed["second"] == (N_GENERATED - INTERRUPT_AFTER) * per_condition


def test_noise_floor_rows_for_every_seed_and_language(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    zero = df[(df["vector_type"] == "sentiment") & (df["strength"] == 0.0)]
    for layer in LAYERS:
        got = {(int(s), lang) for s, lang in zip(zero[zero["layer"] == layer]["seed"], zero[zero["layer"] == layer]["lang"])}
        assert got == {(s, lang) for s in SEEDS for lang in LANGS}


def test_controls_present(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    counts = df["vector_type"].value_counts()
    assert counts["random"] == N_RANDOM
    per_type = N_PROMPTING // (2 * len(PLACEMENTS))
    for vt in ("prompt_pos_system", "prompt_neg_system", "prompt_pos_user", "prompt_neg_user"):
        assert counts[vt] == per_type, vt
    assert set(df[df["vector_type"] == "random"]["layer"]) == {RANDOM_LAYER}


def test_measure_columns_filled_and_in_range(tmp_path, resumed) -> None:
    df = _generated(pd.read_csv(tmp_path / "results.csv"))
    for col in MEASURE_COLUMNS:
        assert df[col].notna().all(), col
    for col in ("sentiment_a_mean", "sentiment_b_mean"):
        assert df[col].between(-1, 1).all()
    for col in ("arabic_share", "distinct_bigram_share"):
        assert df[col].between(0, 1).all()
    assert (df["n_samples"] == N_PROMPTS).all()


def test_split_half_rows_hold_stability_only(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    sh = df[df["vector_type"] == "split_half"]
    for col in STABILITY_COLUMNS:
        assert sh[col].notna().all()
        assert sh[col].between(-1, 1).all()
    assert sh[["strength", "seed", "lang"]].isna().all().all()


def test_samples_one_line_per_prompt_per_condition(tmp_path, resumed) -> None:
    samples = _samples(tmp_path)
    assert len(samples) == N_GENERATED * N_PROMPTS
    keyed = {(condition_key(s), s["prompt_id"]) for s in samples}
    assert len(keyed) == len(samples)
    assert all(isinstance(s["text"], str) for s in samples)


def test_rows_are_ordered_by_key_columns(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    assert list(df.columns[: len(KEY_COLUMNS)]) == list(KEY_COLUMNS)


def test_random_skipped_when_layer_unset(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data, no_network) -> None:
    cfg = dataclasses.replace(cfg, random_direction=dataclasses.replace(cfg.random_direction, layer=None))
    _run(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data)
    df = pd.read_csv(tmp_path / "results.csv")
    assert "random" not in set(df["vector_type"])


def test_grid_for_reads_smoke_and_full_from_config(cfg) -> None:
    full = grid_for(cfg, smoke=False)
    assert (full.layers, full.strengths, full.seeds, full.n_prompts) == (cfg.layers, cfg.strengths, cfg.seeds, None)
    smoke = grid_for(cfg, smoke=True)
    assert (smoke.layers, smoke.strengths, smoke.seeds, smoke.n_prompts) == (
        cfg.smoke.layers,
        cfg.smoke.strengths,
        cfg.smoke.seeds,
        cfg.smoke.n_prompts,
    )


def test_random_has_no_strength_zero_rows(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv")
    assert (df[df["vector_type"] == "random"]["strength"] != 0.0).all()


def test_prompting_types_follow_config(cfg) -> None:
    assert prompting_types(cfg) == {
        "prompt_pos_system": ("pos", "system"),
        "prompt_neg_system": ("neg", "system"),
        "prompt_pos_user": ("pos", "user"),
        "prompt_neg_user": ("neg", "user"),
    }


def test_user_placement_puts_instruction_in_user_turn(tmp_path, resumed) -> None:
    samples = _samples(tmp_path)
    user = [s for s in samples if s["vector_type"] == "prompt_neg_user" and s["lang"] == "en"]
    system = [s for s in samples if s["vector_type"] == "prompt_neg_system" and s["lang"] == "en"]
    neg_en = "respond in a very negative tone ."
    assert user and all(s["system"] is None and neg_en in s["user_message"] for s in user)
    assert system and all(s["system"] == neg_en and neg_en not in s["user_message"] for s in system)


def test_every_row_carries_one_fingerprint_and_git_sha(tmp_path, resumed) -> None:
    df = pd.read_csv(tmp_path / "results.csv", dtype={"fingerprint": str})
    assert df["fingerprint"].nunique() == 1
    assert df["fingerprint"].str.fullmatch(r"[0-9a-f]{12}").all()
    assert df["git_sha"].notna().all()


def test_fingerprint_tracks_method_settings_not_grid(cfg, data) -> None:
    base = fingerprint(cfg, data)
    assert fingerprint(cfg, data) == base
    # Grid and the hand-set random layer may grow between passes without invalidating rows.
    grown = dataclasses.replace(cfg, layers=cfg.layers + (99,), strengths=cfg.strengths + (3.0,),
                                random_direction=dataclasses.replace(cfg.random_direction, layer=7))  # fmt: skip
    assert fingerprint(grown, data) == base
    hotter = dataclasses.replace(cfg, sampling=dataclasses.replace(cfg.sampling, temperature=1.3))
    assert fingerprint(hotter, data) != base
    other_data = dataclasses.replace(data, pairs=data.pairs[:-1])
    assert fingerprint(cfg, other_data) != base


def test_changed_method_settings_do_not_reuse_old_rows(
    tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data, no_network
) -> None:
    _run(tmp_path, tiny_model, tiny_tokenizer, cfg, grid, data)
    hotter = dataclasses.replace(cfg, sampling=dataclasses.replace(cfg.sampling, temperature=1.3))
    _run(tmp_path, tiny_model, tiny_tokenizer, hotter, grid, data)
    df = pd.read_csv(tmp_path / "results.csv", dtype={"fingerprint": str})
    assert df["fingerprint"].nunique() == 2
    assert len(_generated(df)) == 2 * N_GENERATED


def test_model_id_defaults_to_dev_for_smoke_and_main_for_full(cfg) -> None:
    assert model_id_for(cfg, smoke=True, choice=None) == cfg.model.dev_id
    assert model_id_for(cfg, smoke=False, choice=None) == cfg.model.id
    # The T4 smoke run uses the smoke grid on the main model.
    assert model_id_for(cfg, smoke=True, choice="main") == cfg.model.id
    assert model_id_for(cfg, smoke=False, choice="dev") == cfg.model.dev_id


def test_followup_config_adds_layers_under_the_first_pass_fingerprint() -> None:
    """config.followup.yaml changes only what the fingerprint leaves out, so its run extends the first pass."""
    root = Path(__file__).resolve().parents[1]
    first = load_config(root / "config.yaml")
    followup = load_config(root / "config.followup.yaml")
    assert set(first.layers) < set(followup.layers)
    assert followup.random_direction.layer not in (None, first.random_direction.layer)
    same = dataclasses.replace(
        followup,
        layers=first.layers,
        random_direction=dataclasses.replace(followup.random_direction, layer=first.random_direction.layer),
    )
    assert same == first
    assert fingerprint(followup, load_data(followup)) == fingerprint(first, load_data(first))
