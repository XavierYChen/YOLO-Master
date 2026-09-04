# Overhead measurement plan

The admission runner measures a diagnostic upper bound on each model family with the same image, resolution, dtype, device, warmup, and iteration count.

- Baseline sample: one inference-only model forward.
- Instrumented sample: temporary hook registration, the same forward, detached routing snapshots copied to CPU, and hook removal.
- Synchronization: before and after every timed CUDA/MPS sample.
- Reported statistics: median, p95, min, max, median delta, and median ratio.
- Excluded from timing: JSON serialization and PNG rendering.

This is a smoke-stage measurement, not proof of the separate P1 “training slowdown <10%” objective. That objective requires a trained workload, a persistent instrumentation mode, repeated runs, discarded outliers, and a locked power/thermal state. The raw admission number must be reported even when it is unfavorable.
