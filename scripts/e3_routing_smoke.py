"""Run the E3 admission smoke for MoE, MoT, and Latent routing families.

This is an architecture/instrumentation smoke, not an accuracy benchmark. Models are
built from YAML with random initialization, one real image is forwarded, routing
snapshots are validated, and the cost of collecting a snapshot is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILES = {
    "moe": ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml",
    "mot": ROOT / "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
    "latent": ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
}


@dataclass(frozen=True)
class LayerEvidence:
    """Serializable evidence for one routed layer."""

    layer_name: str
    module_type: str
    family: str
    num_experts: int
    top_k: int
    expert_usage: list[float]
    dominant_share: float
    normalized_entropy: float
    normalized_gini: float
    dead_experts: list[int]
    tensor_shape: list[int]
    aux_loss: dict[str, Any]


def normalized_entropy(values: list[float]) -> float:
    """Return entropy normalized to [0, 1]."""
    clean = [max(float(value), 0.0) for value in values]
    total = sum(clean)
    if len(clean) <= 1 or total <= 0:
        return 0.0
    probabilities = [value / total for value in clean]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    return min(max(entropy / math.log(len(probabilities)), 0.0), 1.0)


def normalized_gini(values: list[float]) -> float:
    """Return finite-sample-normalized Gini in [0, 1]."""
    clean = sorted(max(float(value), 0.0) for value in values)
    count = len(clean)
    total = sum(clean)
    if count <= 1 or total <= 0:
        return 0.0
    gini = (2.0 * sum((index + 1) * value for index, value in enumerate(clean)) / (count * total)) - (
        count + 1.0
    ) / count
    return min(max(gini * count / (count - 1), 0.0), 1.0)


def percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def family_matches(family: str, module_type: str) -> bool:
    """Select the routed modules that are the subject of each family profile."""
    name = module_type.lower()
    if family == "latent":
        return "latent" in name
    if family == "mot":
        return "mot" in name
    if family == "moe":
        return "moe" in name and "latent" not in name
    return False


def module_family(module: Any) -> str:
    """Return the routed family's canonical marker."""
    snapshot = getattr(module, "last_routing_snapshot", {})
    if isinstance(snapshot, dict) and snapshot.get("family"):
        return str(snapshot["family"]).lower()
    kind = getattr(module, "_routing_aux_kind", None)
    if kind:
        return str(kind).lower()
    module_path = type(module).__module__.lower()
    return next((name for name in ("latent", "mot", "moe") if name in module_path), "unknown")


def aux_status(module: Any) -> dict[str, Any]:
    """Describe eval-time aux state without presenting zero as training evidence."""
    snapshot = getattr(module, "last_routing_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    balance = float(getattr(module, "balance_loss_coeff", 0.0) or 0.0)
    z_loss = float(getattr(module, "router_z_loss_coeff", 0.0) or 0.0)
    configured = balance > 0.0 or z_loss > 0.0
    return {
        "status": "configured_inactive_eval" if configured and not module.training else "not_configured",
        "observed": float(snapshot.get("aux_loss", 0.0) or 0.0),
        "balance_loss_coeff": balance,
        "router_z_loss_coeff": z_loss,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco8.yaml", help="dataset YAML used to resolve a real validation image")
    parser.add_argument("--image", type=Path, help="optional image override; defaults to the first validation image")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/e3_routing_smoke/results")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or mps")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    return parser


def _select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _time_call(torch: Any, device: Any, callback: Any, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        callback()
    _sync(torch, device)
    samples = []
    for _ in range(iterations):
        _sync(torch, device)
        start = time.perf_counter()
        callback()
        _sync(torch, device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _resolve_image(data: str, override: Path | None) -> tuple[Path, dict[str, str]]:
    """Resolve one real validation image from a dataset YAML."""
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data, autodownload=True)
    dataset_root = Path(str(dataset["path"])).resolve()
    evidence = {"config": data, "root_name": dataset_root.name}
    for key in ("train", "val"):
        if key not in dataset:
            continue
        value = dataset[key][0] if isinstance(dataset[key], (list, tuple)) else dataset[key]
        value_path = Path(str(value)).resolve()
        try:
            evidence[key] = str(value_path.relative_to(dataset_root)).replace("\\", "/")
        except ValueError:
            evidence[key] = value_path.name
    if override is not None:
        image = override.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        return image, evidence
    values = dataset.get("val")
    values = values if isinstance(values, (list, tuple)) else [values]
    candidates = []
    for value in values:
        path = Path(str(value))
        if path.is_dir():
            candidates.extend(
                item for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        elif path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no validation images resolved from {data!r}")
    return min(candidates).resolve(), evidence


def _layer_evidence(summary: Any, collapse: Any, heatmap: Any, module: Any) -> LayerEvidence:
    usage = [float(value) for value in summary.expert_usage]
    return LayerEvidence(
        layer_name=summary.layer_name,
        module_type=summary.module_type,
        family=module_family(module),
        num_experts=summary.num_experts,
        top_k=summary.top_k,
        expert_usage=usage,
        dominant_share=max(usage),
        normalized_entropy=normalized_entropy(usage),
        normalized_gini=normalized_gini(usage),
        dead_experts=list(collapse.dead_experts),
        tensor_shape=list(heatmap.probabilities.shape),
        aux_loss=aux_status(module),
    )


def _validate_layers(layers: list[LayerEvidence]) -> list[str]:
    errors = []
    if not layers:
        return ["no matching routed layers were captured"]
    for layer in layers:
        prefix = layer.layer_name
        if layer.num_experts <= 1:
            errors.append(f"{prefix}: num_experts must be > 1")
        if len(layer.expert_usage) != layer.num_experts:
            errors.append(f"{prefix}: expert_usage length does not match num_experts")
        if not all(math.isfinite(value) and value >= 0 for value in layer.expert_usage):
            errors.append(f"{prefix}: expert_usage has invalid values")
        if not math.isclose(sum(layer.expert_usage), 1.0, rel_tol=0.0, abs_tol=1e-4):
            errors.append(f"{prefix}: expert_usage does not sum to 1")
        if not 0.0 <= layer.normalized_entropy <= 1.0:
            errors.append(f"{prefix}: entropy is outside [0, 1]")
        if not 0.0 <= layer.normalized_gini <= 1.0:
            errors.append(f"{prefix}: Gini is outside [0, 1]")
    return errors


def _profile_run(family: str, args: argparse.Namespace, torch: Any, device: Any) -> dict[str, Any]:
    from tools.routing_interpreter import _load_batch, _load_model
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    config = PROFILES[family]
    model = _load_model(config, device, half=False)
    batch = _load_batch(args.image, args.imgsz, device, torch.float32)
    interpreter = RoutingInterpreter(model)

    captured = interpreter.capture_routing(batch)
    modules = {name or "<root>": module for name, module in model.named_modules()}
    summaries = {item.layer_name: item for item in interpreter.collect_layer_summaries(heatmaps=captured)}
    collapse = interpreter.detect_routing_collapse(heatmaps=captured)
    selected = {name: heatmap for name, heatmap in captured.items() if module_family(modules[name]) == family}
    layers = [
        _layer_evidence(summaries[name], collapse[name], heatmap, modules[name])
        for name, heatmap in selected.items()
        if name in summaries and name in collapse
    ]
    errors = _validate_layers(layers)

    def baseline_forward() -> None:
        with torch.no_grad():
            model(batch)

    def captured_forward() -> None:
        interpreter.capture_routing(batch)

    baseline = _time_call(torch, device, baseline_forward, args.warmup, args.iterations)
    instrumented = _time_call(torch, device, captured_forward, args.warmup, args.iterations)
    delta = instrumented["median_ms"] - baseline["median_ms"]
    overhead = {
        "method": "end_to_end_snapshot_collection",
        "scope": "hook registration + forward + detached CPU routing snapshot",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "baseline": baseline,
        "instrumented": instrumented,
        "median_delta_ms": delta,
        "median_ratio": instrumented["median_ms"] / baseline["median_ms"] if baseline["median_ms"] else None,
        "note": "Upper-bound diagnostic overhead; not steady-state production routing cost.",
    }
    return {
        "family": family,
        "status": "passed" if not errors else "failed",
        "config": str(config.relative_to(ROOT)).replace("\\", "/"),
        "initialization": "random",
        "captured_routed_layers_total": len(captured),
        "matching_layers": [asdict(layer) for layer in layers],
        "validation_errors": errors,
        "overhead": overhead,
    }


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": gpu,
        "git_commit": git_sha,
    }


def _save_figure(profiles: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(len(profiles), 2, figsize=(14, 4.2 * len(profiles)), squeeze=False)
    for row, profile in enumerate(profiles):
        layers = profile.get("matching_layers", [])
        left, right = axes[row]
        if not layers:
            left.text(0.5, 0.5, profile.get("error", "No routing captured"), ha="center", va="center", wrap=True)
            right.axis("off")
            continue
        max_experts = max(layer["num_experts"] for layer in layers)
        matrix = np.full((len(layers), max_experts), np.nan)
        for index, layer in enumerate(layers):
            matrix[index, : layer["num_experts"]] = layer["expert_usage"]
        image = left.imshow(
            matrix,
            vmin=0.0,
            vmax=max(1.0 / max_experts, np.nanmax(matrix)),
            cmap="Blues",
            aspect="auto",
        )
        left.set_title(f"{profile['family'].upper()} expert load (random init)")
        left.set_xlabel("Expert index")
        left.set_ylabel("Routed layer")
        left.set_yticks(range(len(layers)), [layer["layer_name"] for layer in layers])
        left.set_xticks(range(max_experts))
        figure.colorbar(image, ax=left, fraction=0.025, pad=0.02)
        positions = np.arange(len(layers))
        right.barh(positions - 0.18, [layer["normalized_entropy"] for layer in layers], 0.36, label="Entropy")
        right.barh(positions + 0.18, [layer["normalized_gini"] for layer in layers], 0.36, label="Gini")
        right.set_yticks(positions, [layer["layer_name"] for layer in layers])
        right.set_xlim(0.0, 1.0)
        right.set_title("Routing balance diagnostics")
        right.legend()
    figure.suptitle("E3 admission smoke: MoE / MoT / Latent", fontsize=16)
    figure.tight_layout()
    figure.savefig(output / "routing_snapshot.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    for profile in profiles:
        layers = profile.get("matching_layers", [])
        if not layers:
            continue
        columns = max(layer["num_experts"] for layer in layers)
        matrix = np.full((len(layers), columns), np.nan)
        for index, layer in enumerate(layers):
            matrix[index, : layer["num_experts"]] = layer["expert_usage"]
        family_figure, axis = plt.subplots(figsize=(max(7.0, columns * 0.8), max(3.5, len(layers) * 0.5 + 2)))
        family_image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(columns), [f"E{index}" for index in range(columns)])
        axis.set_yticks(range(len(layers)), [layer["layer_name"] for layer in layers])
        axis.set_xlabel("Expert")
        axis.set_ylabel("Routed layer")
        axis.set_title(f"E3 routing snapshot - {profile['family'].upper()} expert usage")
        family_figure.colorbar(family_image, ax=axis, label="Mean routing probability")
        family_figure.tight_layout()
        family_figure.savefig(output / f"{profile['family']}_expert_usage.png", dpi=160, bbox_inches="tight")
        plt.close(family_figure)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix not in {".log"} and path.name != "manifest.sha256.json":
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    _write_json(output / "manifest.sha256.json", {"algorithm": "sha256", "files": files})


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be >= 1")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "false")

    import torch

    args.image, dataset = _resolve_image(args.data, args.image)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device(torch, args.device)
    print(f"E3 routing smoke | device={device} | image={args.image}", flush=True)

    profiles = []
    for family in args.families:
        print(f"[{family}] building and capturing routing...", flush=True)
        try:
            profile = _profile_run(family, args, torch, device)
        except Exception as error:  # Keep remaining families observable in one run.
            profile = {
                "family": family,
                "status": "failed",
                "config": str(PROFILES[family].relative_to(ROOT)).replace("\\", "/"),
                "initialization": "random",
                "matching_layers": [],
                "validation_errors": [f"{type(error).__name__}: {error}"],
            }
        profiles.append(profile)
        print(f"[{family}] {profile['status']}", flush=True)

    passed = all(profile["status"] == "passed" for profile in profiles) and len(profiles) == len(args.families)
    command = (
        f"python scripts/e3_routing_smoke.py --data {args.data} --device auto --imgsz "
        f"{args.imgsz} --warmup {args.warmup} --iterations {args.iterations}"
    )
    snapshot = {
        "schema_version": "e3.routing_snapshot.v1",
        "purpose": "architecture and instrumentation admission smoke",
        "accuracy_claim": False,
        "dataset": args.data,
        "input": {"image": args.image.name, "imgsz": args.imgsz},
        "seed": args.seed,
        "profiles": profiles,
    }
    environment = _environment(torch, device)
    input_record = {
        "image": args.image.name,
        "dataset": dataset,
        "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "imgsz": args.imgsz,
        "batch_size": 1,
        "dtype": "float32",
    }
    resolved_config = {
        family: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for family, path in PROFILES.items()
        if family in args.families
    }
    summary = {
        "status": "passed" if passed else "failed",
        "families_requested": args.families,
        "families_passed": [profile["family"] for profile in profiles if profile["status"] == "passed"],
        "admission_rule": "all requested families build, expose matching routes, and emit valid normalized usage",
        "warnings": [
            "Random initialization is not evidence of model accuracy or trained expert specialization.",
            "Overhead includes diagnostic hook setup and CPU snapshot copies; "
            "use it as an upper-bound tool-cost measurement.",
        ],
    }
    _write_json(args.output / "routing_snapshot.json", snapshot)
    _write_json(args.output / "route_stats.json", snapshot)
    _write_json(args.output / "environment.json", environment)
    _write_json(args.output / "input.json", input_record)
    _write_json(args.output / "config.resolved.json", resolved_config)
    _write_json(args.output / "summary.json", summary)
    (args.output / "command.txt").write_text(command + "\n", encoding="utf-8")
    _save_figure(profiles, args.output)
    _manifest(args.output)
    print(f"Result: {summary['status']} | evidence={args.output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
