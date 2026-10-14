# Slice 01: config, results schema and tiny-model fixture (walking skeleton)

**Spec:** docs/steering_experiment_spec.md#3-setup (also §5.3, §6, §8) · **Issue:** [#3](https://github.com/Muhanad-husn/steering-xling/issues/3)
**Depends on:** none

## Goal
`load_config("config.yaml")` returns a typed config holding every setting the experiment uses, and `seed_everything(n)` seeds torch, numpy and `random`. `src/schema.py` fixes the `results.csv` columns and the resume key. The test suite can build a tiny, randomly initialised Qwen2-architecture model on CPU with no downloads.

## Acceptance criterion
- **Given** the repo's `config.yaml`,
- **When** `load_config` reads it,
- **Then** it returns a frozen dataclass whose model IDs, layers (4, 8, 12, 16, 20, 24), strengths (-2 … 2, 0 included), seeds (3), sampling (temperature 0.7, top_p 0.9, 60 new tokens), scorer IDs and paths match spec §3 and §5.3. A missing key raises a `ValueError` naming that key. With network access blocked, the `tiny_model` fixture builds a `Qwen2ForCausalLM` on CPU from a config of at most 4 layers and d_model at most 64.

## Mechanism
- Library: `pyyaml` with standard-library `dataclasses`. No config framework.
- Tiny model: `transformers.Qwen2Config(...)` and `Qwen2ForCausalLM(config)` under a fixed seed, plus a tiny tokenizer built in-process (a `PreTrainedTokenizerFast` over a word-level vocabulary from `tokenizers`, with a minimal chat template). `conftest.py` sets `HF_HUB_OFFLINE=1`.
- Scorer IDs were checked on 14 Oct 2026 and all resolve. The DistilBERT ID is written in its canonical form, `distilbert/distilbert-base-uncased-finetuned-sst-2-english`: the old ID redirects to the same model, so this is a rename, not a substitution. Qwen2.5-1.5B has 28 layers, so the layer grid is valid.
- `src/schema.py` defines the CSV key `(model, layer, strength, seed, lang, vector_type)`, the measure columns, and the `vector_type` values `sentiment`, `random`, `prompt_pos`, `prompt_neg` and `split_half` (see the open question in the README).

## Files
```aeo-independence
slice: 01-config-skeleton
edits: requirements.txt
creates: config.yaml
creates: src/__init__.py
creates: src/config.py
creates: src/schema.py
creates: tests/conftest.py
creates: tests/test_config.py
creates: docs/tdd-evidence/01-config-skeleton.md
```

## Out of scope
Model forward passes beyond building the fixture; loading scorers; the Colab notebook.
