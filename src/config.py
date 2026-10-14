"""Typed, frozen view of config.yaml, and seeding for torch, numpy and random."""

import dataclasses
import random
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    id: str
    dev_id: str
    dtype: str


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class ExtractionConfig:
    batch_size: int


@dataclass(frozen=True)
class StabilityConfig:
    n_splits: int
    seed: int


@dataclass(frozen=True)
class RandomDirectionConfig:
    layer: int | None
    seed: int


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int
    ci: float
    seed: int


@dataclass(frozen=True)
class ScorersConfig:
    a: str
    b: dict[str, str]


@dataclass(frozen=True)
class PathsConfig:
    pairs: Path
    prompts: Path
    reference: Path
    instructions: Path
    results_csv: Path
    samples: Path
    figures: Path


@dataclass(frozen=True)
class SmokeConfig:
    layers: tuple[int, ...]
    strengths: tuple[float, ...]
    seeds: tuple[int, ...]
    n_prompts: int


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    layers: tuple[int, ...]
    strengths: tuple[float, ...]
    seeds: tuple[int, ...]
    languages: tuple[str, ...]
    sampling: SamplingConfig
    extraction: ExtractionConfig
    stability: StabilityConfig
    random_direction: RandomDirectionConfig
    bootstrap: BootstrapConfig
    scorers: ScorersConfig
    paths: PathsConfig
    smoke: SmokeConfig


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in typing.get_args(tp):
            return None
        (inner,) = [a for a in typing.get_args(tp) if a is not type(None)]
        return _coerce(inner, value, where)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"config key '{where}' must be a list")
        (inner, _) = typing.get_args(tp)
        return tuple(_coerce(inner, v, where) for v in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"config key '{where}' must be a mapping")
        return {str(k): str(v) for k, v in value.items()}
    if value is None:
        raise ValueError(f"config key '{where}' must not be empty")
    if tp is Path:
        return Path(value)
    if tp is float:
        return float(value)
    if tp is int and isinstance(value, bool):
        raise ValueError(f"config key '{where}' must be an integer")
    return tp(value)


def _build(cls: Any, raw: Any, prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"config key '{prefix or '<root>'}' must be a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    dotted = lambda name: f"{prefix}.{name}" if prefix else name  # noqa: E731
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ValueError(f"unknown config key '{dotted(unknown[0])}'")
    kwargs = {}
    for name in names:
        if name not in raw:
            raise ValueError(f"missing config key '{dotted(name)}'")
        kwargs[name] = _coerce(hints[name], raw[name], dotted(name))
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _build(Config, raw, "")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
