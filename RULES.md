# Rules

The rules this experiment is held to. Source: `docs/steering_experiment_spec.md`.

1. **Noise floor first.** Every effect reported sits beside the standard deviation at strength 0 over 3 seeds. An effect smaller than that is not claimed.
2. **Controls are not optional.** Every result at the best layer is shown next to the random-direction control and the prompting baseline.
3. **Two scorers.** No sentiment claim rests on scorer A alone. Scorer B (DistilBERT SST-2 for English, CAMeLBERT for Arabic) must point the same way, or the disagreement gets reported.
4. **Side effects are measured alongside the effect.** Perplexity, Arabic-script share, distinct-bigram share and length are recorded for every condition the sentiment is.
5. **Arabic is read, not only scored.** The founder reads the Arabic outputs before any Arabic claim goes into REPORT.md.
6. **Traceability.** Every claim in REPORT.md traces to a row in `results/results.csv` or to a figure regenerated from that file alone.
7. **Limits go first.** Small model, one behaviour, automatic scorers (and their weaknesses on Arabic dialects), one run of the sweep. These are stated in the report's first paragraph.
8. **Reproducible by construction.** All settings live in `config.yaml`, seeds are fixed, versions are pinned in `requirements.txt`, and the sweep resumes from the CSV after a crash.
9. **Honest framing.** The report is dated and called what it is: a cross-lingual steering experiment run this week. A failed or null result gets reported as one.
10. **Kill line.** Stop and rethink the method, rather than tuning further, if either of these holds:
    - split-half cosine similarity stays below 0.5 at every layer, so the direction is noise;
    - at no layer and strength does the English sentiment shift beat both the noise floor and the random-direction control while perplexity stays under 2x baseline.

    A null Arabic transfer is not a kill. It is an answer to Q2.
