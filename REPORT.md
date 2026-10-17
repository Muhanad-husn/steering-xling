# Does an English sentiment vector steer Arabic? A small experiment, 17 Oct 2026

An experiment run this week, on one model, one sweep, one behaviour.

**Limits first.**
- **The model is small:** `Qwen/Qwen2.5-1.5B-Instruct`. Its Arabic is weak even unsteered: the founder rated 2 of 5 baseline Arabic answers good or acceptable.
- **One behaviour** (sentiment), steered by one method: a mean-difference vector added at one layer.
- **Automatic scorers:**
  - Two automatic scorers stand in for a reader. Scorer A is multilingual XLM-R. Scorer B is SST-2 DistilBERT for English and CAMeLBERT for Arabic.
  - Neither is reliable on dialect, and scorer B reads broken or list-shaped English as negative.
- **Model-generated data:** a Claude Haiku agent wrote the contrast pairs, prompts, reference sentences and their Modern Standard Arabic (MSA) translations. The founder reviewed the Arabic.
- **A single sweep run**, 3 seeds. One random direction, at one layer.
- **Small Arabic reading:** the Arabic claims rest on the founder reading 85 blind samples, 5 per condition. That is a reading, not a measurement.

## Question
Can a sentiment direction built only from 60 English sentence pairs steer the model when the prompt is in Arabic? How large is the effect against a noise floor, a random direction and plain prompting, and what does it cost in fluency?

## Setup
- **Vector.** For each layer L, the vector is the mean over the positive sentences of the residual stream after layer L, minus the same mean over the negative sentences. Each sentence's activations are averaged over its tokens.
- **Hook.** Steering adds `alpha × mean_norm_L × unit_vector_L` at every position. `mean_norm_L` leaves out the first token, which is an attention sink.
- **Sweep.**
  - Layers 4 to 24 in steps of 4.
  - Strengths -2 to 2.
  - 3 seeds, 30 English and 30 MSA prompts, 60 new tokens, temperature 0.7.
- **Controls.**
  - Noise floor: the std at strength 0 over the 3 seeds.
  - A random unit direction at layer 12, all strengths.
  - Tone requested in the prompt, in the system prompt and, separately, in the user message.
- **Fluency.** Perplexity on 20 neutral reference sentences per language. Also the Arabic-script share and the distinct-bigram share.
- **Run.** One Colab T4. Fingerprint `b7be7e59b0aa`, code `e0ba522`. All numbers below come from `results/results.csv`; the run record is in `logs/2026-10-17-t4-full-1.5b/`.

**Noise floors** (std at strength 0, 3 seeds):

| language | scorer A | scorer B | baseline perplexity |
|---|---|---|---|
| English | 0.03 (mean 0.35) | 0.17 (mean 0.42) | 6.0 |
| Arabic | 0.04 (mean 0.48) | 0.05 (mean 0.29) | 12.3 |

**Split-half stability.** Over 20 random half-splits of the pairs, the cosine between the two halves' vectors is 0.990 to 0.999 at every layer (figure 4). The direction is stable, far above the 0.5 kill threshold.

## Results

### English: steering works at strength ±1 in layers 12 and 16, but a random direction also moves sentiment
Shifts are measured against strength 0 at the same layer. The noise floors are A 0.03 and B 0.17.

| condition | shift A | shift B | perplexity |
|---|---|---|---|
| sentiment L12 +1 | +0.17 | +0.27 | 9.0 |
| sentiment L16 +1 | +0.17 | +0.42 | 7.9 |
| random L12 +1 | -0.38 | -0.32 | 31.8 |
| sentiment L12 +0.5 | +0.07 | +0.19 | 6.5 |
| random L12 +0.5 | +0.17 | +0.23 | 7.6 |
| sentiment L12 -1 | -0.35 | -0.98 | 8.9 |
| sentiment L16 -1 | -0.45 | -1.10 | 9.1 |
| random L12 -1 | -0.79 | -1.33 | 21.9 |
| prompt, positive, user message | +0.22 | +0.44 | n/a |
| prompt, negative, user message | -0.80 | -1.07 | n/a |

- **Kill line (RULES.md).** The rule is that some cell's English shift must beat the noise floor and the random direction, with perplexity under 2x baseline. L12 +1 is the only like-for-like cell that passes: +0.27 on B, against a floor of 0.17, a random shift of -0.32 and a perplexity of 1.5x baseline. The founder accepted this literal pass on 17 Oct 2026. It is a narrow pass.
- **The random direction is not neutral.**
  - At ±0.25 and ±0.5 it moves sentiment with the sign of the strength, as far as the sentiment vector does.
  - At ±1 and beyond, both directions break the output, and broken output scores negative.
  - So below strength 1 this experiment cannot tell the sentiment vector from any large push on the residual stream.
- **Negative steering is mostly format, not tone.** At L12 -1, scorer B moves -0.98 but scorer A moves only -0.35. The sample below is a neutral numbered list, not negative prose.
- **The best layer is not the controlled one.** L16 +1 is the strongest English cell under 2x perplexity (B +0.42, perplexity 7.9), but there is no random control at L16. Spec §5.4 leaves rerunning the control there to the founder.
- **Prompting beats steering.** Asking for the tone in the user message moves English further than any usable steering cell, at no perplexity cost.
- **Breakdown.** Beyond ±1 the output breaks at every layer. Layers 4 and 24 break by ±0.5, and layer 20 by ±1 (figures 1 and 3).

### Arabic: by the founder's reading, the English vector does not carry tone into Arabic
Noise floors are A 0.04 and B 0.05. Baseline perplexity is 12.3 and Arabic share 0.99.

| condition | shift A | shift B | perplexity | Arabic share | founder: good or acceptable / tone pos-neu-neg-unreadable (of 5) |
|---|---|---|---|---|---|
| strength 0 | 0 | 0 | 12.3 | 0.99 | 2 / 1-3-0-1 |
| sentiment L12 +0.5 | -0.06 | +0.02 | 14.2 | 0.98 | 2 / 3-1-0-1 |
| sentiment L16 +0.5 | 0.00 | +0.02 | 14.1 | 0.96 | 4 / 3-2-0-0 |
| sentiment L12 +1 | +0.05 | +0.22 | 21.9 | 1.00 | 1 / 2-2-0-1 |
| sentiment L16 +1 | +0.15 | +0.25 | 21.4 | 0.99 | 1 / 2-1-0-2 |
| sentiment L12 -0.5 | -0.24 | -0.20 | 14.3 | 0.60 | 1 / 0-0-0-5 |
| sentiment L16 -0.5 | -0.27 | -0.24 | 15.0 | 0.41 | 0 / 0-0-0-5 |
| random L12 +0.5 | +0.06 | +0.11 | 14.9 | 0.99 | 3 / 3-0-1-1 |
| random L12 -0.5 | -0.31 | -0.21 | 15.6 | 0.75 | 2 / 0-3-0-2 |
| prompt, negative, user message | -0.65 | -0.59 | n/a | 1.00 | 3 / 0-0-5-0 |
| prompt, positive, user message | +0.15 | +0.21 | n/a | 0.98 | 4 / 3-1-0-1 |

- **Negative steering pushes the model out of Arabic.** At -0.5 the Arabic share falls to 0.60 (L12) and 0.41 (L16). The founder read all 10 of those samples as unreadable. The L12 ones answer in English, in French, or in Arabic mixed with Latin and Cyrillic words. The scorers' negative shift here is the language breaking, not tone.
- **Positive steering at +1 moves the scorers above the floor** (B +0.22 to +0.25), at about 1.8x perplexity. The founder read 2 of 5 positive at each layer against 1 of 5 at baseline, and only 1 of 5 good or acceptable. That is no clear tone shift, and quality falls.
- **The random direction matches positive steering at +0.5** (3 of 5 read positive in both).
- **Prompting carries tone into Arabic.** A negative instruction in the user message read negative in 5 of 5 (in the system prompt, 4 of 5), and the scorers agree (A -0.65, B -0.59).

### What did not work
- No negative steering in Arabic at any strength: the language goes before the tone.
- Strength ±2 breaks the output at every layer.
- **Figures 2 and 5 are drawn at strength 2.** `plots.py` uses the largest strength, where every layer's output is broken, so the transfer ratios in figure 2 (from -4.6 to 6.0) mean nothing. The tables above use strengths under 2x perplexity instead. **Pending the founder's decision** on which strength these figures should use.
- **Time.** The sweep took 5.6 hours on the T4, against the spec's target of under 2 hours.

## Example generations (seed 0, the bread prompt in English, founder's notes for Arabic)
1. **English, strength 0:** "A bakery prepares bread by mixing flour, water, yeast, and salt into dough, allowing it to rise through fermentation before shaping and baking until golden brown."
2. **English, L12 +1:** "Bread baking is an art that involves many steps, starting with selecting high-quality flour, … nurturing the dough, and taking care of delicious bread." Warmer and readable.
3. **English, L16 +1:** "Bread baking is an art and science that involves several steps. First, the dough rises…" Fluent, mildly warmer.
4. **English, L12 -1:** "[1] The ingredients are taken from the flour, yeast, salt and water; [2] these raw materials are mixed into a dough…" A neutral list, not negative, though scorer B calls this condition strongly negative.
5. **English, random L12 -1:** the answer comes out in Chinese, as questions about bread weights. A failure.
6. **English, L4 -2:** whitespace and symbols. A failure.
7. **Arabic, strength 0, bread:** "1. يتم إعداد مزيج من الأعشاب الطازجة والصلصة الحارة…" Founder: "the domain is correct, the answer is wrong".
8. **Arabic, L16 +0.5, kitchen table:** "هي طاولة مطبخ أنيقة وقوية، تتميز بتصميمها المميز والجذاب." Founder: good, positive.
9. **Arabic, L12 -0.5, bread:** "1. The baker starts with flour and water. 2. He mixes the ingredients…" Founder: "the quality is good and the Answer is complete but it's in ENGLISH".
10. **Arabic, negative prompt in the user message, spring garden:** "الحديقة في أوائل الربيع كانت خالية من الحياة، حيث لم تظهر أي نور أو نشاط." Founder: good, negative.

## Answers
- **Q1, does steering shift sentiment in English with dose?** Partly. The shift grows with strength inside ±1 at layers 12 and 16. The positive side passes the controls only at L12 +1, and the random direction moves just as much at smaller strengths.
- **Q2, does the English vector transfer to Arabic?** Not by the founder's reading. The scorers see a small positive shift at +1, but the reader does not, and negative steering breaks the language first.
- **Q3, where does fluency break?** Beyond ±1 at layers 12 and 16, and earlier elsewhere. In Arabic, output leaves the language at -0.5.
- **Q4, steering or prompting?** On this model, prompting is stronger in both languages and costs no fluency.

## Next steps (not started; each needs the founder's decision)
- Run random directions at L16, and several random draws at L12, to test whether the pass is specific to sentiment.
- Choose the strength for figures 2 and 5, then redraw them.
- Run on a larger or more Arabic-capable model (Gemma 2 2B, spec §3).

## Figures
All five figures are drawn from `results/results.csv` with `python -m src.plots --csv results/results.csv --out results/figures`.
1. `results/figures/fig1_dose_response.png`: sentiment against strength, per layer.
2. `results/figures/fig2_transfer_ratio.png`: Arabic shift over English shift. Drawn at strength 2; see "What did not work".
3. `results/figures/fig3_trade_off.png`: sentiment shift against perplexity increase. A few broken cells stretch the x-axis.
4. `results/figures/fig4_split_half_stability.png`: split-half cosine per layer.
5. `results/figures/fig5_steering_vs_baselines.png`: steering, random direction and prompting at L12. Drawn at strength 2.
