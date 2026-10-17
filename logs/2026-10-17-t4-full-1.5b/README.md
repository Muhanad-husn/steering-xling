# T4 full sweep on the 1.5B model, 17 Oct 2026 (phase 3)

- **Command:** `notebooks/colab.ipynb` with `GRID = "full"` and `MODEL = "main"`.
- **Code and settings:** commit `e0ba522`, fingerprint `b7be7e59b0aa` on every row.
- **Machine:** Google Colab, Tesla T4. Model `Qwen/Qwen2.5-1.5B-Instruct`, float16.
- **Time:** `real 336m35s` in `run.log`.
- **Grid:** layers 4 to 24 in steps of 4, strengths -2 to 2 (with ±0.25 and ±0.5), seeds 0 to 2, en and ar. Random direction at layer 12 only. Four prompting baselines.
- **Files:** `results.csv` (402 rows), `samples.jsonl`, `run.log`, `arabic-outputs.md`, plus the founder's reading files listed below.

## Checks
| check | result |
|---|---|
| one fingerprint, one code commit | yes, `b7be7e59b0aa`, `e0ba522` |
| one row per key | yes, 0 duplicates |
| NaN perplexity | only the 6 split-half rows, which have no perplexity by design |
| sentiment in [-1, 1] | yes, B from -1.00 to 1.00 |
| split-half cosine | 0.990 to 0.999 by layer (min 0.977 at layer 24); far above 0.5 |
| resume rerun adds nothing | **not verified** |
| tests at branch head | 136 passed (CPU, this session) |

## English results (sentiment B, mean of 3 seeds)
Noise floor: std at strength 0 over 3 seeds = **0.17** (baseline mean 0.42). Baseline perplexity 6.03, so the 2x line is 12.06.

| condition | -1 | -0.5 | +0.5 | +1 |
|---|---|---|---|---|
| sentiment L12 | -0.56 (ppl 8.9) | 0.07 (6.6) | 0.62 (6.5) | 0.70 (9.0) |
| sentiment L16 | -0.67 (9.1) | 0.10 (6.5) | 0.53 (6.4) | 0.84 (7.9) |
| random L12 | -0.90 (21.9) | -0.13 (7.2) | 0.65 (7.6) | 0.10 (31.8) |

Prompting: pos system 0.82, pos user 0.86, neg system -0.33, neg user -0.65.

## Kill-line check (for the founder's decision, nothing tuned)
Random rows exist at layer 12 only, so a like-for-like comparison is possible only there.

- **L12 +1:** shift +0.27 from baseline, above the 0.17 floor. The random direction at +1 moves -0.32 (wrong way, ppl 31.8). Perplexity 9.0, under 2x. **Passes on a literal reading.** Margin over the floor is small.
- **L12 -1:** shift -0.98, but random moves further (-1.33). Random's perplexity is 21.9, over 2x, so its shift is partly broken output. Fails "beats random" on magnitude.
- **L12 ±0.5:** random moves as far or further than sentiment in both directions. Fails.
- **Random is not neutral.** At ±0.25 and ±0.5 it moves sentiment B in the sign of the strength, much like the vector does. One random direction is one draw; the cosine between it and the sentiment vector is not recorded.
- **L16 +1** (shift +0.42, ppl 7.9) is the strongest English cell under 2x, but has no random control at its layer.

## Arabic (scorer, seed means) and the founder's reading
Scorer B at strength 0 is 0.29, Arabic share 0.99. At -0.5 the Arabic share falls to 0.60 (L12) and 0.41 (L16); at -1 to 0.13 and 0.42. Positive strengths keep Arabic.

The founder rated 85 blind samples (seed 0, prompts q01, q07, q13, q19, q25) for quality and tone: `reading-sheet.md`, `reading-key.md`, `reading-form.html`, `arabic-ratings.csv`, `arabic-ratings-with-key.csv`, `arabic-ratings-by-condition.csv`.

| condition (5 each) | good or acceptable | pos / neu / neg / unreadable |
|---|---|---|
| baseline | 2 | 1 / 3 / 0 / 1 |
| L12 -0.5 | 1 | 0 / 0 / 0 / 5 |
| L12 -0.25 | 4 | 0 / 4 / 1 / 0 |
| L12 +0.25 | 1 | 1 / 3 / 0 / 1 |
| L12 +0.5 | 2 | 3 / 1 / 0 / 1 |
| L12 +1 | 1 | 2 / 2 / 0 / 1 |
| L16 -0.5 | 0 | 0 / 0 / 0 / 5 |
| L16 -0.25 | 3 | 2 / 1 / 1 / 1 |
| L16 +0.25 | 2 | 2 / 2 / 1 / 0 |
| L16 +0.5 | 4 | 3 / 2 / 0 / 0 |
| L16 +1 | 1 | 2 / 1 / 0 / 2 |
| random L12 -0.5 | 2 | 0 / 3 / 0 / 2 |
| random L12 +0.5 | 3 | 3 / 0 / 1 / 1 |
| prompt neg system | 3 | 0 / 0 / 4 / 1 |
| prompt neg user | 3 | 0 / 0 / 5 / 0 |
| prompt pos system | 5 | 3 / 2 / 0 / 0 |
| prompt pos user | 4 | 3 / 1 / 0 / 1 |

- Negative prompting reads negative in 9 of 10. No steering condition reads negative in more than 1 of 5.
- -0.5 is unreadable at both layers, mostly English or Chinese output. The founder marked English answers "very bad" with the note "it's in ENGLISH".
- Positive steering is no clearer than the random direction at +0.5.
- Five samples per condition: a reading, not a measurement.

## Open
- **Kill line:** founder decision on the literal L12 +1 pass, given random's behaviour.
- Random control at other layers, and the random/sentiment cosine, are not measured.
- Resume on the T4 path is not verified.
- Figures not yet generated from this CSV.
