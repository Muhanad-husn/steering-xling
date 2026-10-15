"""Slice 03: steering hook, random direction and seeded generation (spec 5.2 to 5.4)."""

import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from src.config import SamplingConfig
from src.steer import generate, random_unit_vector, steering

LAYER = 1
MEAN_NORM = 3.0
SAMPLING = SamplingConfig(temperature=0.7, top_p=0.9, max_new_tokens=8)


def _logits(model: Qwen2ForCausalLM, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids).logits


@pytest.fixture
def ids(tiny_tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    return tiny_tokenizer("the hotel was good .", return_tensors="pt")["input_ids"]


@pytest.fixture
def unit(tiny_model: Qwen2ForCausalLM) -> torch.Tensor:
    return random_unit_vector(tiny_model.config.hidden_size, seed=7)


def test_alpha_zero_leaves_logits_unchanged(tiny_model, ids, unit) -> None:
    base = _logits(tiny_model, ids)
    with steering(tiny_model, LAYER, unit, MEAN_NORM, alpha=0.0):
        steered = _logits(tiny_model, ids)
    assert torch.equal(base, steered)


def test_nonzero_alpha_changes_logits_and_hook_is_removed(tiny_model, ids, unit) -> None:
    base = _logits(tiny_model, ids)
    with steering(tiny_model, LAYER, unit, MEAN_NORM, alpha=1.0):
        steered = _logits(tiny_model, ids)
    assert not torch.allclose(base, steered)
    assert torch.equal(base, _logits(tiny_model, ids))
    assert not tiny_model.model.layers[LAYER]._forward_hooks


def test_hook_adds_scaled_vector_at_every_position(tiny_model, ids, unit) -> None:
    with torch.no_grad():
        base = tiny_model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1]
        with steering(tiny_model, LAYER, unit, MEAN_NORM, alpha=-0.5):
            out = tiny_model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1]
    expected = base + (-0.5 * MEAN_NORM) * unit
    torch.testing.assert_close(out, expected)


def test_hook_removed_when_body_raises(tiny_model, unit) -> None:
    with pytest.raises(RuntimeError):
        with steering(tiny_model, LAYER, unit, MEAN_NORM, alpha=1.0):
            raise RuntimeError("boom")
    assert not tiny_model.model.layers[LAYER]._forward_hooks


def test_random_unit_vector_norm_and_determinism() -> None:
    a = random_unit_vector(32, seed=1234)
    assert a.shape == (32,)
    assert torch.isclose(a.norm(), torch.tensor(1.0))
    assert torch.equal(a, random_unit_vector(32, seed=1234))
    assert not torch.equal(a, random_unit_vector(32, seed=1235))


def test_generate_same_seed_same_tokens(tiny_model, tiny_tokenizer) -> None:
    first = generate(tiny_model, tiny_tokenizer, "describe the city", SAMPLING, seed=0)
    second = generate(tiny_model, tiny_tokenizer, "describe the city", SAMPLING, seed=0)
    assert first.token_ids == second.token_ids
    assert 0 < len(first.token_ids) <= SAMPLING.max_new_tokens
    assert isinstance(first.text, str)


def test_generate_system_instruction_enters_prompt(tiny_model, tiny_tokenizer) -> None:
    plain = generate(tiny_model, tiny_tokenizer, "describe the city", SAMPLING, seed=0)
    toned = generate(
        tiny_model, tiny_tokenizer, "describe the city", SAMPLING, seed=0,
        system="respond very positive tone",
    )
    assert "system" not in plain.prompt
    assert "respond very positive tone" in toned.prompt
