# Cross-lingual activation steering: does an English steering vector work in Arabic?

Project spec for implementation in Claude Code. Author: Muhanad Abulhusn. Drafted 14 Oct 2026.

## 1. Purpose

A small, complete, honestly reported steering experiment that shows three things:

1. I can extract a steering vector from an open-weight model, apply it, and measure the effect.
2. I measure properly: noise floor, controls, side effects, stated limits.
3. A question that matters for multilingual products: a vector built only from English data, does it also steer the model when the prompt is in Arabic?

The deliverable is a public repo plus a short report, dated, described as an experiment run this week.

## 2. Research question and hypotheses

- **Q1.** Does adding a sentiment direction to the residual stream (the model's main internal channel) shift the sentiment of generated text, in a dose-dependent way?
- **Q2.** Does a direction built only from English sentence pairs shift sentiment for Arabic prompts too, and how much weaker is it?
- **Q3.** At what strength does steering start to damage the output (fluency, language, repetition)?
- **Q4.** How does steering compare with simply asking for the tone in the prompt?

Hypotheses: H1 effect grows with strength and peaks in middle layers; H2 transfer to Arabic is real but weaker; H3 there is a usable window before fluency breaks; H4 prompting is stronger on frontier-style instructions but steering is harder to override.

## 3. Setup

| Item | Choice | Why |
|---|---|---|
| Main model | `Qwen/Qwen2.5-1.5B-Instruct` | Open weights, no licence gate, decent Arabic, fits a free Colab T4 |
| Second model (optional) | `google/gemma-2-2b-it` (licence must be accepted on Hugging Face) | Checks the result is not one model's quirk |
| CPU fallback for development | `Qwen/Qwen2.5-0.5B-Instruct` | Fast smoke tests |
| Behaviour steered | Sentiment, positive vs negative | Can be scored automatically in both languages |
| Scorer A | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual, includes Arabic) | Score = P(positive) minus P(negative), range -1 to 1 |
| Scorer B (cross-check) | English: `distilbert-base-uncased-finetuned-sst-2-english`; Arabic: `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment` | A second, independent scorer so the result does not rest on one classifier |
| Libraries | PyTorch, transformers, plain forward hooks (no TransformerLens needed), pandas, matplotlib | Minimal dependencies |
| Hardware | One GPU with 16 GB (Colab T4 is enough) | |

Check the scorer model names on Hugging Face before coding; replace any that no longer exist with the nearest equivalent and record the change.

> **Checked 15 Oct 2026.** All model and scorer IDs above resolve on Hugging Face. `distilbert-base-uncased-finetuned-sst-2-english` now redirects to `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. This is the same model under its canonical namespace, not a substitution, and `config.yaml` uses the canonical ID. Qwen2.5-1.5B-Instruct has 28 decoder layers, so the §5.3 layer grid is valid.

## 4. Data (stored as JSONL in `data/`)

> **Amended 14 Oct 2026.** The data is no longer hand-written by me. A Claude Haiku agent generates it to the brief below, `tests/test_data.py` checks it mechanically, and I read the Arabic before any Arabic claim (§9 already requires that). I looked for an open dataset first. None has English minimal sentiment pairs alongside neutral English/MSA prompts. The prompting-baseline instructions live in `data/instructions.jsonl`. REPORT.md states this provenance as a limit.

1. **Contrast pairs, English only:** 60 pairs of sentences identical except for sentiment. Example: "The service at the hotel was wonderful." / "The service at the hotel was terrible." Vary topics (food, travel, products, work, weather, films) so the direction is about sentiment, not topic.
2. **Neutral prompts:** 30 English and the same 30 translated into Arabic by me (Modern Standard Arabic). Example: "Write two sentences about a train journey." / "اكتب جملتين عن رحلة بالقطار."
3. **Reference texts for side effects:** 20 neutral sentences per language, used only to measure fluency (perplexity).
4. **Prompting baseline instructions:** "Respond in a very positive tone." / "Respond in a very negative tone." in both languages.

## 5. Method

### 5.1 Extract the steering vector
- For each pair, run both sentences through the model and record the residual stream after every decoder layer, averaged over the sentence's tokens.
- Vector for layer L = mean(positive) minus mean(negative). Store the raw vector, its unit version, and the average residual norm at that layer.

> **Amended 15 Oct 2026.** The average residual norm leaves out each sentence's first token. The phase 3 smoke run on Qwen2.5-0.5B showed that token's norm at about 1600 against about 13 for the rest (an attention sink), so averaging over all tokens made strength 1 roughly 15 times a typical token's norm and every non-zero strength destroyed the output (perplexity in the millions). The founder chose this fix over a median norm or smaller strengths, before any real result was seen. The steering vector itself is unchanged. Run record: `logs/2026-10-15-smoke/`.
- **Stability check:** split the pairs randomly into two halves 20 times, build a vector from each half, and report the cosine similarity between halves per layer. Low similarity means the direction is noise.

### 5.2 Apply it
- Forward hook on decoder layer L that adds `alpha * mean_norm_L * unit_vector_L` to the layer output at every position.
- Strength is relative to the layer's own scale so values are comparable across layers.

### 5.3 Sweep
- Layers: 6 evenly spaced across depth (for a 28-layer model: 4, 8, 12, 16, 20, 24). After the first pass, add finer layers around the best one.
- Strength multipliers: -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2.
- 3 sampling seeds per condition, temperature 0.7, top_p 0.9, 60 new tokens, chat template applied.
- All 60 prompts (30 English, 30 Arabic) per condition.

### 5.4 Controls
- **Noise floor:** strength 0 across the 3 seeds. Report the standard deviation; any effect smaller than this is not claimed.
- **Random direction:** a random unit vector with the same norm, at the best layer and all strengths. If it also shifts sentiment, the result is not specific.
- **Prompting baseline:** no steering, tone requested in the system prompt, and separately in the user message.

> **Amended 16 Oct 2026.** The random-direction layer is set before the full sweep, not after its first pass. In the T4 smoke run on Qwen2.5-1.5B (`logs/2026-10-16-t4-smoke-1.5b/`), layer 12 was the only smoke layer that steered without breaking the output; layer 6 turned every sample into code at -1 and looped at +1. The founder set `random_direction.layer: 12`, so the full sweep writes the random rows in the same run. This was decided from smoke data (one seed, 4 prompts, layers 6 and 12 only), before any full-sweep result. If the first pass shows a different best layer, the report says so and the founder decides whether to rerun the control there.

> **Amended 17 Oct 2026.** After REPORT.md, the founder asked for the open controls to be run. The second pass adds the random direction at layer 16, the strongest English layer, and the finer layers 10, 14 and 18 around layers 12 and 16. The agent picked those three layers. The pass runs from `config.followup.yaml`, which differs from `config.yaml` only in `layers` and `random_direction.layer`. Neither is in the fingerprint, so the new rows join the first pass's 402 under `b7be7e59b0aa`, and `tests/test_run_sweep.py` checks that. `config.yaml` stays the first pass. Several random draws at layer 12 would need a code change, and any change to the code changes the fingerprint. The founder dropped them on 17 Oct 2026, so the report keeps "one random direction" as a limit.

> **Amended 15 Oct 2026.** In the phase 3 smoke run on Qwen2.5-0.5B, the system-prompt instruction did not work: "respond in a very negative tone" produced cheerful English, and the founder read both Arabic prompting outputs as very bad. The founder chose to keep the system-prompt baseline and add a second baseline that puts the instruction at the start of the user message (`prompting.user_template` in `config.yaml`). Both run in every sweep as `prompt_<tone>_system` and `prompt_<tone>_user`, and the report shows both, so neither is chosen after the results are seen. The stronger of the two is the harder bar for steering. This was decided before any real (1.5B) result.
- **Optional, phase 2, directional ablation:** project the sentiment direction out of the residual stream, then give explicitly emotional prompts ("Describe the best day of your life"). Does sentiment flatten? This is the "remove part of the model and see what breaks" test.

### 5.5 Measures per condition
- Sentiment score (scorer A and B), mean and 95% bootstrap interval.
- Fluency: perplexity of the steered model on the reference texts.
- Language retention: share of Arabic-script characters in outputs for Arabic prompts.
- Repetition: share of distinct 2-word sequences.
- Output length.

## 6. Outputs

- `results/results.csv`: one row per model, fingerprint, layer, strength, seed, language, vector type, with all measures and the git commit.

> **Amended 15 Oct 2026.** `fingerprint` is a 12-character hash of every setting, data file and source file that shapes a row, leaving out the grid (layers, strengths, seeds, languages, the hand-set random layer). The sweep resumes on it, so rows made under old settings or old code are never reused, and figures never mix them. `git_sha` records the commit but is not part of the key.
- `results/samples.jsonl`: every prompt and generation, for inspection.
- `results/figures/`:
  1. Dose-response: sentiment against strength, one line per layer, English and Arabic side by side.
  2. Transfer ratio: Arabic effect divided by English effect, per layer.
  3. Trade-off: sentiment shift against perplexity increase, to show the usable window.
  4. Split-half stability per layer.
  5. Steering vs prompting vs random direction at the best layer.
- `REPORT.md`, at most two pages: question, setup, results with figures, what did not work, limits, next steps.
- 10 hand-picked example generations (English and Arabic), including failures.

## 7. Repo layout

```
steering-xling/
  README.md            how to run, in 5 lines
  REPORT.md
  config.yaml          model, layers, strengths, seeds, paths
  data/                pairs.jsonl, prompts.jsonl, reference.jsonl, instructions.jsonl
  src/
    extract.py         activations and vectors, stability check
    steer.py           hook, generation
    score.py           scorers, perplexity, language, repetition
    run_sweep.py       loops over the config, writes results.csv
    plots.py
  tests/               unit tests for the hook, vector maths and scoring
  notebooks/colab.ipynb  one-click run on Colab
```

## 8. Requirements for Claude Code

- Python 3.12+, typed, minimal dependencies, all settings in `config.yaml`, fixed seeds, library versions pinned in `requirements.txt`.

> **Amended 17 Oct 2026.** Python 3.12+, not 3.11+. The pinned `numpy==2.5.4` publishes no wheels for Python 3.11, so the pinned set cannot install there. The founder agreed the change. No result depends on it.

- Unit tests that run on CPU with a tiny randomly initialised model: the hook changes the output only when the strength is non-zero, the vector has the right shape, and the scores stay in range.
- The sweep can resume after a crash (skip conditions already in the CSV).
- Every figure is regenerated from `results.csv` alone.
- Each result in REPORT.md comes with its numbers and the noise floor next to it.

## 9. Done when

- [ ] The sweep runs end to end on Qwen 1.5B on a T4 in under 2 hours.
- [ ] Every claim in REPORT.md traces to a row or figure.
- [ ] Noise floor, random direction and prompting baseline are all reported.
- [ ] The Arabic outputs have been read and checked by me, not only scored.
- [ ] Limits stated: small model, one behaviour, automatic scorers (and their known weaknesses on Arabic dialects), single run of the sweep.

## 10. Plan

| Step | Time |
|---|---|
| Write the data files (pairs, prompts, Arabic translations) | 1.5 h |
| Claude Code builds src and tests, smoke test on 0.5B on CPU | 2 h |
| Full sweep on Colab | 1 to 2 h |
| Read the outputs, write the report | 1.5 h |

A first version is realistic in one long day. Phase 2 (Gemma, ablation) comes after.

## 11. How to present it

- Date it and call it what it is: "a cross-lingual steering experiment I ran this week".
- Lead with the question that matters to them: does steering built in English survive in Arabic?
- Show the measurement discipline, not just the effect: noise floor, controls, side effects.
- State the limits in the first paragraph of the report, not at the end.
