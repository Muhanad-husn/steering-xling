"""Steering hook, random-direction control and seeded generation (spec sections 5.2 to 5.4).

The hook is a plain forward hook on decoder layer L (`model.model.layers[L]`). It adds
`alpha * mean_norm * unit_vector` to the layer's output at every position, so its effect
appears in `hidden_states[L + 1]`, the same residual the vector was extracted from. transformers 5 records `hidden_states` with its
own forward hook on each layer; ours is prepended so that record shows the steered residual.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import SamplingConfig, seed_everything


@contextmanager
def steering(
    model: PreTrainedModel,
    layer: int,
    unit_vector: torch.Tensor,
    mean_norm: float,
    alpha: float,
) -> Iterator[None]:
    """Add `alpha * mean_norm * unit_vector` to decoder layer `layer`'s output while open."""
    shift = (alpha * float(mean_norm)) * unit_vector.detach()

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + shift.to(device=hidden.device, dtype=hidden.dtype)
        return (steered, *output[1:]) if isinstance(output, tuple) else steered

    handle = model.model.layers[layer].register_forward_hook(hook, prepend=True)
    try:
        yield
    finally:
        handle.remove()


def random_unit_vector(d_model: int, seed: int) -> torch.Tensor:
    """Gaussian direction of norm 1, float32 on CPU, fixed by `seed`."""
    gen = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(d_model, generator=gen), dim=0)


@dataclass(frozen=True)
class Generation:
    prompt: str  # the chat-templated prompt text
    token_ids: list[int]  # new tokens only
    text: str  # new tokens decoded, special tokens dropped


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    sampling: SamplingConfig,
    seed: int,
    system: str | None = None,
) -> Generation:
    """Chat-templated sampling, seeded for torch, numpy and random before each call.

    `system` carries the tone instruction for the prompting baseline.
    """
    messages = [{"role": "system", "content": system}] if system is not None else []
    messages.append({"role": "user", "content": user_prompt})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    seed_everything(seed)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_new_tokens=sampling.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[0, enc["input_ids"].shape[1] :].tolist()
    return Generation(prompt=prompt, token_ids=new, text=tokenizer.decode(new, skip_special_tokens=True))
