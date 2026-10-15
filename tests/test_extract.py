"""Slice 02: steering-vector extraction and split-half stability (CPU, tiny model, no downloads)."""

import math
from collections.abc import Mapping

import torch
from transformers import PreTrainedTokenizerBase, Qwen2ForCausalLM

from src.extract import extract_vectors, split_half_stability

N_SPLITS = 7
SEED = 3
PAIRS: list[Mapping[str, str]] = [
    {"positive": "the food was wonderful .", "negative": "the food was terrible ."},
    {"positive": "i loved the hotel", "negative": "i hated the hotel"},
    {"positive": "the film was good", "negative": "the film was bad"},
    {"positive": "great weather in the city this morning", "negative": "awful weather in the city this morning"},
    {"positive": "happy work", "negative": "sad work"},
    {"positive": "the service was very good !", "negative": "the service was very bad !"},
    {"positive": "a wonderful train journey", "negative": "a terrible train journey"},
    {"positive": "the kitchen table was great", "negative": "the kitchen table was awful"},
]


def test_shapes_and_norms(tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase) -> None:
    out = extract_vectors(tiny_model, tiny_tokenizer, PAIRS)
    n_layers = tiny_model.config.num_hidden_layers
    d_model = tiny_model.config.hidden_size
    assert out.raw.shape == (n_layers, d_model)
    assert out.unit.shape == (n_layers, d_model)
    assert out.mean_norm.shape == (n_layers,)
    assert torch.allclose(out.unit.norm(dim=-1), torch.ones(n_layers), atol=1e-5)
    assert torch.isfinite(out.mean_norm).all()
    assert (out.mean_norm > 0).all()


def test_swapping_pos_neg_negates_raw(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    swapped = [{"positive": p["negative"], "negative": p["positive"]} for p in PAIRS]
    a = extract_vectors(tiny_model, tiny_tokenizer, PAIRS)
    b = extract_vectors(tiny_model, tiny_tokenizer, swapped)
    assert torch.allclose(a.raw, -b.raw, atol=1e-5)


def test_batching_does_not_change_result(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    whole = extract_vectors(tiny_model, tiny_tokenizer, PAIRS)
    small = extract_vectors(tiny_model, tiny_tokenizer, PAIRS, batch_size=3)
    assert torch.allclose(whole.raw, small.raw, atol=1e-5)
    assert torch.allclose(whole.mean_norm, small.mean_norm, atol=1e-5)


def test_split_half_cosines(tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase) -> None:
    cos = split_half_stability(tiny_model, tiny_tokenizer, PAIRS, n_splits=N_SPLITS, seed=SEED)
    assert cos.shape == (tiny_model.config.num_hidden_layers, N_SPLITS)
    assert all(-1.0 <= c <= 1.0 and not math.isnan(c) for c in cos.flatten().tolist())


def test_split_half_same_seed_identical(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    a = split_half_stability(tiny_model, tiny_tokenizer, PAIRS, n_splits=N_SPLITS, seed=SEED)
    b = split_half_stability(tiny_model, tiny_tokenizer, PAIRS, n_splits=N_SPLITS, seed=SEED)
    c = split_half_stability(tiny_model, tiny_tokenizer, PAIRS, n_splits=N_SPLITS, seed=SEED + 1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_mean_norm_excludes_each_sentence_first_token(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    """Spec 5.1, amended 15 Oct 2026: the first token's attention-sink norm would set the scale."""
    sentences = [p["positive"] for p in PAIRS] + [p["negative"] for p in PAIRS]
    total = torch.zeros(tiny_model.config.num_hidden_layers)
    count = 0
    with torch.no_grad():
        for s in sentences:  # one at a time: no padding, so position 0 is the first token
            ids = tiny_tokenizer(s, return_tensors="pt")["input_ids"]
            hs = torch.stack(tiny_model(input_ids=ids, output_hidden_states=True).hidden_states[1:])
            total += hs[:, 0, 1:].norm(dim=-1).sum(dim=-1)
            count += ids.shape[1] - 1
    out = extract_vectors(tiny_model, tiny_tokenizer, PAIRS, batch_size=3)
    assert torch.allclose(out.mean_norm, total / count, atol=1e-4)
