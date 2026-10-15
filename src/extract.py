"""Steering-vector extraction and split-half stability (spec section 5.1).

Activations come from `output_hidden_states=True`: `hidden_states[L + 1]` is the residual
stream after decoder layer L (`hidden_states[0]` is the embedding output). Each sentence is
averaged over its real tokens, padding masked out. No hooks are needed to read them.

`mean_norm` leaves out each sentence's first token (spec 5.1, amended 15 Oct 2026). In Qwen2.5
that token is an attention sink whose residual norm is about 100 times the others, and
including it made strength 1 several times a typical token's norm.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

Pair = Mapping[str, str]


@dataclass(frozen=True)
class Vectors:
    """Per decoder layer L (row L): raw vector, unit vector and mean residual norm."""

    raw: torch.Tensor  # [n_layers, d_model], mean(positive) - mean(negative)
    unit: torch.Tensor  # [n_layers, d_model], raw / ||raw||
    mean_norm: torch.Tensor  # [n_layers], mean per-token residual norm over all sentences


@torch.no_grad()
def _sentence_means(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[str],
    batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ([n_sent, n_layers, d] token-averaged residuals, [n_layers] summed norms, n_tokens).

    The norms and n_tokens skip each sentence's first token; the averaged residuals keep it.
    """
    device = next(model.parameters()).device
    step = batch_size or len(sentences)
    means: list[torch.Tensor] = []
    norm_sum: torch.Tensor | None = None
    n_tokens = 0
    for start in range(0, len(sentences), step):
        enc = tokenizer(list(sentences[start : start + step]), return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)  # [b, t, 1]
        counts = mask.sum(dim=1)  # [b, 1]
        # hidden_states[L + 1] is the residual after layer L.
        layers = torch.stack(out.hidden_states[1:], dim=0).to(torch.float32)  # [L, b, t, d]
        means.append(((layers * mask).sum(dim=2) / counts).permute(1, 0, 2))  # [b, L, d]
        real = enc["attention_mask"].to(torch.float32)  # [b, t]
        first = (real.cumsum(dim=1) == 1).to(torch.float32) * real  # first real token, any padding side
        norm_mask = real - first
        tok_norms = (layers.norm(dim=-1) * norm_mask).sum(dim=(1, 2))  # [L]
        norm_sum = tok_norms if norm_sum is None else norm_sum + tok_norms
        n_tokens += int(norm_mask.sum().item())
    assert norm_sum is not None, "no sentences given"
    return torch.cat(means, dim=0).cpu(), norm_sum.cpu(), n_tokens


def _unit(raw: torch.Tensor) -> torch.Tensor:
    return F.normalize(raw, dim=-1)


def _pair_means(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[Pair],
    batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pos [n, L, d], neg [n, L, d], mean token norm [L])."""
    sentences = [p["positive"] for p in pairs] + [p["negative"] for p in pairs]
    means, norm_sum, n_tokens = _sentence_means(model, tokenizer, sentences, batch_size)
    n = len(pairs)
    return means[:n], means[n:], norm_sum / n_tokens


def extract_vectors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[Pair],
    batch_size: int | None = None,
) -> Vectors:
    """Steering vector per decoder layer. `batch_size=None` runs all sentences in one batch."""
    pos, neg, mean_norm = _pair_means(model, tokenizer, pairs, batch_size)
    raw = pos.mean(dim=0) - neg.mean(dim=0)
    return Vectors(raw=raw, unit=_unit(raw), mean_norm=mean_norm)


def split_half_stability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[Pair],
    n_splits: int,
    seed: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Cosine between vectors built from two random halves of the pairs, [n_layers, n_splits].

    Activations are computed once; each split only re-averages them. The same seed gives the
    same splits. With an odd pair count the extra pair goes to the second half.
    """
    pos, neg, _ = _pair_means(model, tokenizer, pairs, batch_size)
    diffs = pos - neg  # [n, L, d]; mean of diffs == mean(pos) - mean(neg)
    n = diffs.shape[0]
    if n < 2:
        raise ValueError("split-half stability needs at least 2 pairs")
    half = n // 2
    gen = torch.Generator().manual_seed(seed)
    cosines: list[torch.Tensor] = []
    for _ in range(n_splits):
        perm = torch.randperm(n, generator=gen)
        a = diffs[perm[:half]].mean(dim=0)
        b = diffs[perm[half:]].mean(dim=0)
        cosines.append(F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0))  # [L]
    return torch.stack(cosines, dim=1)
