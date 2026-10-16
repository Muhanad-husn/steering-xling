# T4 smoke run on the 1.5B model, 16 Oct 2026 (phase 3, issue #17)

- **Command:** `notebooks/colab.ipynb` with `GRID = "smoke"` and `MODEL = "main"`, which runs `python -m src.run_sweep --config config.yaml --grid smoke --model main`.
- **Code and settings:** commit `6158e98` (main), fingerprint `b7be7e59b0aa` on every row.
- **Machine:** Google Colab, Tesla T4 (15 GB), torch 2.14.1+cu130, transformers 5.19.0, Python 3.13. Model `Qwen/Qwen2.5-1.5B-Instruct`, float16.
- **Time:** 297 s wall clock (`real 4m56.9s` in `run.log`) for 20 generating conditions × 4 prompts (80 generations), including model and scorer loading.
- **Grid:** layers 6 and 12, strengths -1, 0 and 1, seed 0, en and ar, both prompt placements. `random_direction.layer` is null, so there are no random rows.
- **Files:** `results.csv` (22 rows), `samples.jsonl` (80 lines), `run.log` and `arabic-outputs.md`, as written by the notebook.

## Earlier attempts in the same session
`run.log` opens with two failed attempts, each about 8 s. Importing `transformers` failed with `RuntimeError: operator torchvision::nms does not exist`. Colab's preinstalled `torchvision 0.26.0` is built for torch 2.11, and the pinned torch 2.14.1 replaced it. The third attempt ran. The notebook does not yet handle this; see "Open" below.

## Checks
| check | result |
|---|---|
| float16 path: NaN or inf perplexity | none (all 20 values finite) |
| scorer labels | matched by name; `score.py` raises on missing labels and the run finished |
| one fingerprint across all rows | yes, `b7be7e59b0aa` |
| one row per key (model, fingerprint, layer, strength, seed, lang, vector type) | yes, 0 duplicates |
| sentiment in [-1, 1] | yes, A from -0.53 to 0.81, B from -1.00 to 1.00 |
| resume rerun adds nothing | **not verified**: the run was not repeated on Colab |
| tests in this repo at `6158e98` | 135 passed (CPU, this session) |

## Results (one seed, 4 prompts: not a result, no noise floor)

| layer | strength | lang | sent A | sent B | perplexity | Arabic share |
|---|---|---|---|---|---|---|
| 6 | -1 | en | 0.09 | -0.50 | 9.1 | 0.00 |
| 6 | 0 | en | 0.50 | 0.65 | 6.0 | 0.00 |
| 6 | +1 | en | 0.17 | 0.27 | 12.9 | 0.00 |
| 12 | -1 | en | -0.35 | -1.00 | 8.9 | 0.00 |
| 12 | +1 | en | 0.66 | 0.52 | 9.0 | 0.00 |
| 6 | -1 | ar | 0.07 | 0.18 | 51.2 | 0.00 |
| 6 | 0 | ar | 0.81 | 0.77 | 12.3 | 1.00 |
| 6 | +1 | ar | 0.06 | 0.02 | 34.9 | 1.00 |
| 12 | -1 | ar | 0.07 | 0.11 | 44.4 | 0.12 |
| 12 | +1 | ar | 0.57 | 0.69 | 21.9 | 1.00 |

Strength 0 rows are identical at both layers, as expected.

| baseline | lang | sent A | sent B |
|---|---|---|---|
| pos, system | en | 0.56 | 1.00 |
| neg, system | en | -0.53 | -0.99 |
| pos, user | en | 0.73 | 0.99 |
| neg, user | en | -0.44 | -0.50 |
| pos, system | ar | 0.76 | 0.64 |
| neg, system | ar | 0.08 | -0.28 |
| pos, user | ar | 0.79 | 0.77 |
| neg, user | ar | -0.06 | -0.06 |

Prompting rows carry the strength 0 perplexity (6.0 en, 12.3 ar), because perplexity is measured on the reference sentences without the instruction.

**Split-half cosine:** 0.999 (min 0.998) at layer 6, 0.998 (min 0.995) at layer 12. Far above the 0.5 kill threshold.

## What it showed
- **English steering at layer 12.** Strength -1 moves English negative (A 0.50 to -0.35, B 0.65 to -1.00) with perplexity 1.5x baseline. But 2 of 4 samples are code or JSON rather than prose, so part of the B shift is the broken-output effect noted on 15 Oct. +1 gives repetitive but readable positive text (A 0.66), perplexity 1.5x.
- **Layer 6 is broken in both directions.** At -1 every English and Arabic sample turns into Python code. At +1 the text loops.
- **Arabic -1 leaves Arabic.** The Arabic share is 0.00 at layer 6 and 0.12 at layer 12; outputs are code, English or Chinese.
- **Prompting on 1.5B.** Unlike the dev model, the negative instruction now works in English in both placements (system B -0.99, user B -0.50). In Arabic the scorers show a small move only (neg system B -0.28, neg user A and B -0.06).
- **Against dev run 3 (0.5B):** baseline Arabic perplexity is lower (12.3 vs 27.1) and the Arabic baseline reads better (see below). The steering pattern is the same: layer 12 works in English, and Arabic does not hold up at ±1. The English negative prompt failure was a limit of the small model; it is fixed at 1.5B.

## Figures
`python -m src.plots --csv logs/2026-10-16-t4-smoke-1.5b/results.csv --out <scratch>` fails, as the issue expected: `ValueError: cannot pick the layer: random control found at []; pass --layer`. Because `main()` builds all five figures before it saves any, **no figure is written at all**, not only figure 5. Figures 1 to 4 render when called one by one. Not committed: real figures come from the full sweep.

## Founder's reading of the Arabic outputs (16 Oct 2026), word for word

| condition | founder's reading |
|---|---|
| sentiment layer=6 strength=-1.0 q01 | very bad, the model returned some code, and it has nothing to do with the question |
| sentiment layer=6 strength=-1.0 q02 | very bad, the model returned some code, and it has nothing to do with the question |
| sentiment layer=6 strength=-1.0 q03 | very bad, the model returned some code, and it has nothing to do with the question |
| sentiment layer=6 strength=-1.0 q04 | very bad, the model returned some code, and it has nothing to do with the question |
| sentiment layer=6 strength=0.0 q01 | Acceptable, taking the model size and capabilities in considerations |
| sentiment layer=6 strength=0.0 q02 | Acceptable, taking the model size and capabilities in considerations |
| sentiment layer=6 strength=0.0 q03 | Acceptable, taking the model size and capabilities in considerations |
| sentiment layer=6 strength=0.0 q04 | Good, understood the question and answered to its best capabilities |
| sentiment layer=6 strength=1.0 q01 | Bad |
| sentiment layer=6 strength=1.0 q02 | Bad |
| sentiment layer=6 strength=1.0 q03 | Bad |
| sentiment layer=6 strength=1.0 q04 | Bad |
| sentiment layer=12 strength=-1.0 q01 | very bad, the model returned some code, and it has nothing to do with the question |
| sentiment layer=12 strength=-1.0 q02 | very bad, the model returned the response in English and didn't answer correctly even in English |
| sentiment layer=12 strength=-1.0 q03 | very bad, the model returned the response in English & Chinese and didn't answer correctly even in English |
| sentiment layer=12 strength=-1.0 q04 | very bad, the model returned the response in English & Arabic and didn't answer correctly in both |
| sentiment layer=12 strength=0.0 q01 | good |
| sentiment layer=12 strength=0.0 q02 | Acceptable |
| sentiment layer=12 strength=0.0 q03 | Acceptable |
| sentiment layer=12 strength=0.0 q04 | good |
| sentiment layer=12 strength=1.0 q01 | Acceptable |
| sentiment layer=12 strength=1.0 q02 | bad |
| sentiment layer=12 strength=1.0 q03 | Acceptable |
| sentiment layer=12 strength=1.0 q04 | Very bad, couldn't answer |
| prompt_pos_system q01 | good |
| prompt_pos_system q02 | Acceptable |
| prompt_pos_system q03 | Acceptable |
| prompt_pos_system q04 | Good |
| prompt_neg_system q01 | Good |
| prompt_neg_system q02 | Good |
| prompt_neg_system q03 | bad |
| prompt_neg_system q04 | Good |
| prompt_pos_user q01 | Good |
| prompt_pos_user q02 | Acceptable |
| prompt_pos_user q03 | Acceptable |
| prompt_pos_user q04 | Good |
| prompt_neg_user q01 | Bad |
| prompt_neg_user q02 | Good |
| prompt_neg_user q03 | Good |
| prompt_neg_user q04 | good |

(The strength 0 outputs are identical at layers 6 and 12, so those two sets of readings are of the same texts.)

- The founder's reading is of quality: whether the model answered well. Whether the tone is negative is a separate question, and the table above does not answer it.
- **Steering in Arabic:** -1 is unusable at both layers. +1 at layer 12 is mixed (2 acceptable, 1 bad, 1 very bad); layer 6 +1 is bad throughout.
- **Prompting in Arabic:** on 1.5B all four baselines mostly read acceptable or good. That is a change from the dev model, where three of four were bad.

## Open
- **Colab notebook:** cell 4 should remove or replace Colab's `torchvision` after installing the pinned torch, or the first sweep fails at import. A notebook change belongs in its own PR.
- **`plots.py`:** a figure 5 failure stops figures 1 to 4 being saved. Raise this with the founder before the full sweep's first pass, which also has no random rows.
- **Resume** on the T4 path is not verified.
