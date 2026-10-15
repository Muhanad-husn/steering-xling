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

## Founder's reading of the Arabic outputs, run 2 (15 Oct 2026)

- "Strength +0 is better than strength +1 on both layers." The unsteered Arabic output reads better than the +1 output at layers 6 and 12.
- "As for the prompting, both are very bad." Both the positive-tone and the negative-tone baselines produce poor Arabic.
- So the scorers' Arabic shift at layer 12, +1 (0.52 to 0.67 on scorer A) comes with a quality loss a reader sees. It is not usable steering at that strength on the dev model.

## Run 3: both prompt placements and the fingerprint (founder decision, 15 Oct 2026)

- **Code and settings:** commit `8b65049`, fingerprint `98c7ae998ea0` on every row. Same grid and machine. Took 862 s for 20 conditions.
- **Reproducibility:** the steering and system-placement rows match run 2 exactly (same seed, code path unchanged for them).

| baseline | lang | sent A | sent B |
|---|---|---|---|
| pos, system | en | 0.68 | 0.50 |
| neg, system | en | 0.67 | 0.85 |
| pos, user | en | 0.54 | 0.76 |
| neg, user | en | 0.14 | -0.37 |
| pos, system | ar | 0.76 | 0.65 |
| neg, system | ar | 0.62 | 0.50 |
| pos, user | ar | 0.81 | 0.81 |
| neg, user | ar | 0.80 | 0.64 |

- **English:** the user placement makes the negative instruction partly work. In 2 of 4 samples the text turns negative.
- **Arabic:** neither placement moves sentiment by the scorers. The founder's reading of the run 3 Arabic is pending.

## Founder's reading of the Arabic prompting outputs, run 3 (15 Oct 2026)

| baseline | founder's reading |
|---|---|
| pos, system | "really bad" |
| neg, system | "bad" |
| pos, user | "for such small model, it's good" |
| neg, user | "bad" |

- The user placement is the better one for Arabic on the dev model, but only the positive tone reads well. The negative tone fails in both placements.
- The scorers put every Arabic prompting row between 0.50 and 0.81. The reading agrees that the negative instruction did not work in Arabic.
- This is the dev model with one seed. The T4 smoke run on the 1.5B model decides whether this is a limit of model size.
