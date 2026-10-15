# Smoke run, 15 Oct 2026 (phase 3)

- **Command:** `python -m src.run_sweep --config config.yaml --grid smoke`, branch `claude/great-brown-chrnes`.
- **Machine:** cloud container, 4 CPU cores, 15 GB RAM, no GPU. Model `Qwen/Qwen2.5-0.5B-Instruct`, float32.
- **Time:** 391 s for 16 conditions × 4 prompts (64 generations), plus model and scorer loading.
- **Resume:** a second run wrote no rows or samples and exited in 22 s.
- **Grid:** layers 6 and 12, strengths -1, 0 and 1, seed 0, en and ar. `random_direction.layer` is null, so there are no random rows.
- **Files:** `results.csv` (18 rows) and `samples.jsonl` (64 lines), as written by the sweep.

## Earlier attempts the same day
1. The network policy refused the Hugging Face storage hosts (`cas-server.xethub.hf.co`, then `us.aws.cdn.hf.co`). The founder allowed them.
2. Scorer A failed to load. `cardiffnlp/twitter-xlm-roberta-base-sentiment` ships only `sentencepiece.bpe.model`, and `sentencepiece` was not installed. It is now pinned in `requirements.txt` (0.2.2).

## What it showed
- **Pipeline works end to end.** Every measure column is filled, the split-half rows are written, and the prompting rows are present.
- **Strength ±1 destroys the model at both layers.** Perplexity is 3.6e6 to 1.3e7 against 8.5 (en) and 27 (ar) at strength 0. Outputs are mixed-script garbage, and the Arabic share falls from 1.0 to about 0.01.
- **Likely cause: the mean-norm scale.** The first token's residual norm at layer 6 is about 1600, while other tokens are about 13 (an attention-sink outlier). `mean_norm` averages over all tokens, so strength 1 adds roughly 15 times a typical token's norm. This is a method question for the founder, and nothing was tuned.
- **Split-half cosine** is 0.997 (layer 6) and 0.994 (layer 12), well above the 0.5 kill threshold.
- **Prompting baseline:** `prompt_neg` scores as positive as `prompt_pos` with scorer A (en 0.67 vs 0.68). Not yet read by a human.
- **Scorer B (English)** gives -0.97 to -0.99 on the garbage output, so a broken model can look like a strong negative shift. This is a reason to keep perplexity next to every shift.

The files above are now in `run1-full-mean-norm/`.

## Run 2: mean_norm without the first token (founder chose option 1)

- **Change:** `mean_norm` leaves out each sentence's first token (`src/extract.py`, spec §5.1 amended). Same grid and machine as run 1.
- **Time:** 736 s. Steered outputs are now full-length text, not early-stopping garbage.
- **Files:** `run2-first-token-excluded/`.

| layer | strength | lang | sent A | sent B | perplexity | Arabic share |
|---|---|---|---|---|---|---|
| 6 | -1 | en | -0.20 | -0.22 | 11.5 | 0.00 |
| 6 | 0 | en | 0.34 | 0.64 | 8.5 | 0.00 |
| 6 | +1 | en | 0.00 | 0.02 | 203.4 | 0.00 |
| 12 | -1 | en | -0.53 | -0.99 | 15.2 | 0.00 |
| 12 | +1 | en | 0.77 | 1.00 | 15.9 | 0.00 |
| 6 | -1 | ar | 0.30 | 0.36 | 43.1 | 1.00 |
| 6 | 0 | ar | 0.52 | 0.44 | 27.1 | 1.00 |
| 6 | +1 | ar | 0.27 | 0.16 | 230.0 | 0.98 |
| 12 | -1 | ar | -0.07 | -0.09 | 45.0 | 0.99 |
| 12 | +1 | ar | 0.67 | 0.53 | 60.1 | 0.77 |

- **Reading, smoke only.** One seed and 4 prompts, so there is no noise floor yet and nothing here is a result.
  - Layer 12 moves English sentiment in both directions, with perplexity under 2x baseline.
  - Arabic moves the same way at layer 12, but perplexity is about 2.2x baseline, and at +1 the Arabic share falls to 0.77.
  - Layer 6 at +1 is too strong (perplexity about 24x).
- Nobody has read the Arabic outputs yet.
