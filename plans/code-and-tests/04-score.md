# Slice 04: scorers and side-effect measures

**Spec:** docs/steering_experiment_spec.md#55-measures-per-condition · **Issue:** [#6](https://github.com/Muhanad-husn/steering-xling/issues/6)
**Depends on:** 01-config-skeleton (#3)

## Goal
`score.py` turns the generations for one condition into its measures:
- sentiment from scorer A and from scorer B (picked by language), each as P(positive) − P(negative), reported as a mean with a 95% bootstrap interval;
- perplexity of the model, steered or not, on the reference texts;
- the share of Arabic-script characters;
- the share of distinct bigrams;
- output length.

## Acceptance criterion
- **Given** an injected fake classifier that returns label probabilities in each scorer's own label order (Cardiff: negative, neutral, positive; CAMeLBERT: positive, negative, neutral),
- **When** texts are scored,
- **Then** every sentiment score lies in [-1, 1] and is mapped by label name, not index: a fake that always says "positive" gives +1 under both orders. The bootstrap interval contains the mean and is deterministic in its seed. Arabic-script share and distinct-bigram share lie in [0, 1]: Arabic text gives about 1, Latin text gives 0, and empty text gives 0 without raising. Perplexity on the tiny model is finite and greater than 1.

## Mechanism
- Library: real scorers use `transformers.pipeline("text-classification", top_k=None)`, loaded lazily, with labels resolved through `id2label`. Tests inject a fake callable instead, so nothing is downloaded.
- Perplexity: the model's own loss on the reference texts, under `torch.no_grad`. The caller can pass a steering context, so this slice does not import `steer.py`.
- Arabic share: characters in the Unicode Arabic blocks divided by all letters, ignoring whitespace, digits and punctuation. Distinct bigrams: on whitespace tokens, unique bigrams divided by total bigrams.
- Bootstrap: numpy resampling, with the iteration count and seed from the config.

## Files
```aeo-independence
slice: 04-score
creates: src/score.py
creates: tests/test_score.py
creates: docs/tdd-evidence/04-score.md
depends-on: 01-config-skeleton
```

## Out of scope
Judging how good the scorers are on Arabic dialects (that belongs in the report's limits); aggregating across conditions.
