# Plan

Spec: `docs/steering_experiment_spec.md`. Oracle: the founder reads the results against the checklist below, which comes from spec §9.

## Phases

1. **Data.** Write `data/pairs.jsonl` (60 English contrast pairs across varied topics), `data/prompts.jsonl` (30 neutral prompts in English and the same 30 in MSA Arabic), `data/reference.jsonl` (20 neutral sentences per language), and the prompting-baseline instructions in both languages. The founder writes these.
2. **Code and tests.** Build `extract.py`, `steer.py`, `score.py`, `run_sweep.py`, `plots.py` and `config.yaml`. Write CPU unit tests on a tiny randomly initialised model. Check the scorer model IDs on Hugging Face and record any substitutions.
3. **Smoke test.** Run the end-to-end pipeline on `Qwen/Qwen2.5-0.5B-Instruct` on CPU with a reduced grid.
4. **Full sweep.** Run `Qwen/Qwen2.5-1.5B-Instruct` on a Colab T4, including the noise floor, random direction and prompting baseline, then a finer layer pass around the best layer. Under 2 hours.
5. **Read and report.** The founder reads the samples, Arabic included, and picks 10 examples, failures among them. The five figures are made and REPORT.md is written (two pages at most).
6. **Phase 2 (optional).** Repeat on Gemma-2-2b-it and add the directional-ablation test.

## Founder checklist (the oracle)

- [ ] The sweep runs end to end on Qwen 1.5B on a T4 in under 2 hours.
- [ ] Every claim in REPORT.md traces to a row or figure.
- [ ] Noise floor, random direction and prompting baseline are all reported.
- [ ] The Arabic outputs have been read and checked by the founder, not only scored.
- [ ] Limits are stated in the first paragraph: small model, one behaviour, automatic scorers and their dialect weaknesses, a single sweep run.

## Status

| Phase | State | Score |
|---|---|---|
| 1. Data | todo | 0 of 4 files |
| 2. Code and tests | todo | 0 of 6 modules |
| 3. Smoke test | todo | not run |
| 4. Full sweep | todo | not run |
| 5. Read and report | todo | 0 of 5 checklist items |
| 6. Phase 2 (optional) | todo | not started |
