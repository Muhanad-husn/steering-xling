"""Shared fixtures: a tiny, randomly initialised Qwen2 model and tokenizer, built offline on CPU."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import socket
from collections.abc import Iterator

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

SPECIAL_TOKENS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
WORDS = (
    "the a was is and of to in it this that service hotel food film weather work "
    "wonderful terrible great awful good bad loved hated happy sad write describe "
    "two sentences about train journey kitchen table city morning respond very "
    "positive negative tone system user assistant . , ! ?"
).split()
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}<|im_end|>{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)
TINY_SEED = 0


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + WORDS)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.add_special_tokens(SPECIAL_TOKENS)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        bos_token=None,
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


def build_tiny_model(vocab_size: int) -> Qwen2ForCausalLM:
    torch.manual_seed(TINY_SEED)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=3,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    return build_tiny_tokenizer()


@pytest.fixture
def tiny_model(tiny_tokenizer: PreTrainedTokenizerFast) -> Iterator[Qwen2ForCausalLM]:
    yield build_tiny_model(len(tiny_tokenizer))
