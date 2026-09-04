# E3 Routing Admission Smoke

This report package implements one comparable admission smoke across the **MoE**, **MoT**, and **Latent** families, plus measured routing-instrumentation overhead.

> Scope: architecture and observability only. The three YAML models are randomly initialized. This run proves that each model builds, forwards one real image, exposes its intended router, produces normalized expert usage, and can be inspected. It does **not** claim detection accuracy or trained expert specialization.

## One-command Windows run

Open **Anaconda Prompt**, activate the existing environment, enter the repository, and run:

```bat
reports\e3_routing_smoke\run_smoke.cmd
```

That command installs the local checkout, records the environment, runs all three families, measures overhead, and saves the complete console stream. Python blocks shown on websites are normally saved in a `.py` file and then launched from Anaconda Prompt; `scripts/e3_routing_smoke.py` is exactly that file.

## Fixed experiment design

| Family | Model configuration | Intended routed modules |
| --- | --- | --- |
| MoE | `ultralytics/cfg/models/26/yolo26-master-n.yaml` | `A2C2fMoE` |
| MoT | `ultralytics/cfg/models/26/yolo26-master-mot-n.yaml` | `C2fMoT` |
| Latent | `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` | `LatentMixture` |

All profiles use the first resolved `coco8.yaml` validation image, `imgsz=320`, batch size 1, seed 0, FP32, and the same auto-selected device. Latent's MoE backbone is intentionally excluded from the Latent row so the evidence represents `LatentMixture`, not a second copy of the MoE measurement.

## Admission rule

A family passes only when:

1. its YAML model builds and completes a forward pass;
2. at least one intended routed layer is captured;
3. `expert_usage` is finite, non-negative, has `num_experts` entries, and sums to 1;
4. normalized entropy and Gini are both within `[0, 1]`.

All requested families must pass. Collapse/dead-expert indicators from a single random-initialized image are recorded for diagnosis but are not a failure gate.

## Evidence produced

| File | Meaning |
| --- | --- |
| `results/preflight.log` | Python, YOLO, Torch/CUDA/device environment output |
| `results/full.log` | Complete stdout/stderr from the smoke run — this is the “full log” |
| `results/smoke.log` | Compatibility copy matching the reference report name |
| `results/command.txt` | Exact core reproduction command |
| `results/environment.json` | Machine-readable environment and Git commit |
| `results/input.json` | Input identity, SHA-256, size, batch, and dtype |
| `results/config.resolved.json` | Exact model YAML paths and SHA-256 hashes |
| `results/routing_snapshot.json` | Versioned per-family/per-layer routing evidence |
| `results/route_stats.json` | Compatibility name matching the reference E3 report |
| `results/routing_snapshot.png` | Expert-load heatmaps plus entropy/Gini bars |
| `results/{moe,mot,latent}_expert_usage.png` | Separate family figures matching the reference report layout |
| `results/summary.json` | Overall pass/fail decision and limitations |
| `results/manifest.sha256.json` | Core artifact sizes and SHA-256 integrity hashes (logs excluded because the shell is still writing them) |

Generated `results/` files should be committed after the run so reviewers can verify the exact machine evidence.

## Overhead measurement

For each family the script warms up, then compares 10 synchronized forwards:

- baseline: `model(image)`;
- instrumented: register router hooks, run the same forward, detach routing probabilities, copy the snapshot to CPU, and remove hooks.

The JSON reports median, p95, min, max, delta, and ratio. This is deliberately an **upper-bound diagnostic-tool overhead**, not the model's native routing cost and not a production steady-state benchmark. Overhead is reported, not used as a universal pass threshold, because CPU/GPU synchronization and laptop power state strongly affect a 10-iteration microbenchmark.

## About the old expert-load script

`scripts/ablation_moe_peft_e3_expert_load_viz.py` contains useful Gini and plotting ideas, but it is not the admission runner: it hard-codes a developer's local checkpoint/data paths, focuses on MoLoRA rank allocation, and suppresses validation exceptions. This package instead uses the repository's cross-family `RoutingInterpreter`, explicit family filtering, one schema, and fail-visible results.

## Checklist mapping

| Admission item | Evidence | Completion condition |
| --- | --- | --- |
| 环境安装 | `preflight.log`, `environment.json` | local install and checks succeed |
| 基线/最小任务 | three fixed YAML profiles + one image | all three status values are `passed` |
| 复现命令 | `run_smoke.cmd`, `command.txt` | command is committed |
| 配置文件 | three model YAML paths in snapshot | resolved paths and commit recorded |
| 完整日志 | `preflight.log`, `full.log` | files are non-empty |
| 结果证据 | snapshot JSON/PNG + summary | summary status is `passed` |
| 设计说明 | this README + `SCHEMA.md` | scope and gates documented |
| 风险与降级 | `LIMITATIONS.md` | random-init and timing caveats documented |
| 代码/方案链接 | branch/PR URL | fill after pushing the branch |

Do not mark the three-family E3 admission as complete until `summary.json` says `passed` on the target machine and the generated evidence has been pushed.
