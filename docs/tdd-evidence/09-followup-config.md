# TDD evidence: the second-pass config extends the first pass under its fingerprint (17 Oct 2026)

After REPORT.md, the founder asked for the open controls to be run. The second pass adds the random direction at layer 16 and the finer layers 10, 14 and 18. To compare like with like, its rows must share the first pass's fingerprint (`b7be7e59b0aa`). So `config.followup.yaml` may differ from `config.yaml` only in `layers` and `random_direction.layer`, which the fingerprint leaves out. The test holds it to that.

## Red

`python -m pytest -q tests/test_run_sweep.py -k followup`, with the new test and no `config.followup.yaml`:

```
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/user/steering-xling/config.followup.yaml'
1 failed, 17 deselected in 0.27s
```

The test also fails with the file in place and `sampling.temperature` changed to 0.8.

## Green

```
1 passed, 17 deselected in 0.30s
140 passed in 28.91s   (python -m pytest -q)
```

## Dry run against `results/results.csv` (keys only, no model)

- With `config.yaml`, a resumed full run adds 0 conditions.
- With `config.followup.yaml`, it adds 210 conditions:
  - 54 sentiment at layer 10;
  - 54 sentiment at layer 14;
  - 54 sentiment at layer 18;
  - 48 random at layer 16.

  It also adds three split-half rows. Both runs keep the fingerprint `b7be7e59b0aa`.
