# Smoke run, 15 Oct 2026: blocked before start

- **Command:** `python -m src.run_sweep --config config.yaml --grid smoke` (dev model `Qwen/Qwen2.5-0.5B-Instruct`, CPU, branch `claude/great-brown-chrnes`).
- **Outcome:** no condition ran and no results were written. Only the model's config and tokenizer files (12 MB) downloaded.
- **Cause:** the session's network policy allowed `huggingface.co` but refused the weight-storage hosts:
  - `cas-server.xethub.hf.co` (default Xet download), refused 17 times;
  - `us.aws.cdn.hf.co` (with `HF_HUB_DISABLE_XET=1`), `403 Forbidden`.
- **Next:** allow the `*.hf.co` storage and CDN hosts, then rerun.
