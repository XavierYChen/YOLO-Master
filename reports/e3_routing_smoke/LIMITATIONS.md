# Risks, interpretation, and fallback

- **Random initialization:** this smoke cannot support accuracy, convergence, specialization, or production-quality claims. Follow it with trained-checkpoint validation when checkpoints exist.
- **One image:** a single image is appropriate for interface admission, not distribution-level collapse analysis. Use dataset-mode routing diagnostics for scientific conclusions.
- **Microbenchmark variance:** laptop power mode, thermal state, first-use kernels, and synchronization affect latency. The script reports median and p95 without hiding unfavorable values.
- **Diagnostic overhead scope:** the instrumented timing includes temporary hook registration and a CPU copy. It is an upper bound for generating reviewable evidence.
- **GPU fallback:** `--device auto` uses CUDA, then MPS, then CPU. If CUDA fails, rerun with `--device cpu`; record that downgrade in the report rather than silently presenting it as GPU evidence.
- **Memory:** batch size is fixed at 1 and image size at 320 for a 6 GB laptop GPU. If memory still fails, use `--device cpu`; do not reduce only one family, because comparability would be lost.
