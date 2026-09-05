"""Run the E3 admission smoke for MoE, MoT, and Latent routing families.

This is an architecture/instrumentation smoke, not an accuracy benchmark. Models are
built from YAML with random initialization, one real image is forwarded, native
snapshots or final top-k outputs are validated, and observation cost is measured.
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
import traceback
from datetime import datetime, timezone
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


def validate_vector(values, count, name):
    """Reject invalid evidence instead of silently repairing it."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    if values is None:
        raise ValueError(f"{name}: missing vector")
    values = [float(value) for value in values]
    if len(values) != count or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError(f"{name}: wrong length or non-finite/negative values")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-4):
        raise ValueError(f"{name}: vector does not sum to one")
    return values


class SnapshotCollector:
    """Read leaf snapshots and a strict eval MoE top-k adapter using hooks."""

    def __init__(self, model, family):
        candidates = {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, "last_routing_snapshot") and module_family(module) == family
        }
        self.modules = {
            name: module
            for name, module in candidates.items()
            if not any(other.startswith(name + ".") for other in candidates)
        }
        if not self.modules:
            raise ValueError(f"unsupported: no native snapshot producers for {family}")
        self.family = family
        self.handles = []
        self.records = {}
        self.previous = {}
        self.router_snapshots = {}

    def __enter__(self):
        self.previous = {name: module.last_routing_snapshot for name, module in self.modules.items()}
        try:
            for name, module in self.modules.items():
                if self.family == "moe":
                    if type(module).__name__ != "OptimizedMOEImproved":
                        raise ValueError(f"unsupported MoE adapter: {type(module).__name__}")
                    self.handles.append(module.routing.register_forward_hook(self._moe_hook(name, module)))
                self.handles.append(module.register_forward_hook(self._hook(name)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _moe_hook(self, name, owner):
        def capture(router, inputs, output):
            import torch

            weights, indices = output[:2]
            if weights.ndim != 2 or weights.shape != indices.shape:
                raise ValueError(f"{name}: unsupported top-k tensor layout")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(f"{name}: invalid mixing weights")
            if indices.is_floating_point() or (indices < 0).any() or (indices >= owner.num_experts).any():
                raise ValueError(f"{name}: invalid expert indices")
            if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-4):
                raise ValueError(f"{name}: top-k weights do not sum to one")
            counts = torch.bincount(indices.reshape(-1), minlength=owner.num_experts).float()
            dense = weights.new_zeros((weights.shape[0], owner.num_experts))
            dense.scatter_add_(1, indices.long(), weights)
            self.router_snapshots[name] = {
                "num_experts": owner.num_experts,
                "top_k": weights.shape[1],
                "expert_usage": counts / counts.sum(),
                "mean_router_probs": dense.mean(0),
                "mean_topk_weight": weights.mean(0),
            }

        return capture

    def _hook(self, name):
        def capture(module, inputs, output):
            snapshot = self.router_snapshots.pop(name, None) if self.family == "moe" else module.last_routing_snapshot
            if not snapshot or snapshot is self.previous[name]:
                raise ValueError(f"{name}: missing or stale native snapshot; require MOE_SNAPSHOT_INTERVAL=1")
            self.previous[name] = snapshot
            count = int(snapshot["num_experts"])
            usage = validate_vector(snapshot.get("expert_usage"), count, name)
            probs = snapshot.get("mean_router_probs")
            probs = validate_vector(probs, count, name + ".mean_router_probs") if probs is not None else None
            aux = aux_status(module)
            if self.family == "moe":
                aux["observed"] = None
                aux["status"] = "not_published_eval"
            if aux["observed"] is not None and not math.isfinite(aux["observed"]):
                raise ValueError(f"{name}: non-finite auxiliary loss")
            diagnostics = snapshot.get("finite_diagnostics", {})
            if snapshot.get("finite") is False or diagnostics.get("all_finite") is False:
                raise ValueError(f"{name}: producer reports non-finite routing")
            top_k = int(snapshot["top_k"])
            if not 1 <= top_k <= count:
                raise ValueError(f"{name}: invalid top_k")
            weights = snapshot.get("mean_topk_weight")
            weights = validate_vector(weights, top_k, name + ".mean_topk_weight") if weights is not None else None
            tensor = output[0] if isinstance(output, tuple) else output
            self.records[name] = {
                "layer_name": name,
                "module_type": type(module).__name__,
                "family": self.family,
                "num_experts": count,
                "top_k": top_k,
                "expert_usage": usage,
                "usage_semantics": "topk_selection_share" if self.family == "moe" else "mean_mixture_probability",
                "mean_router_probs": probs,
                "mean_topk_weight": weights,
                "mixing_weights_status": "available" if probs is not None or weights is not None else "unsupported",
                "dominant_share": max(usage),
                "normalized_entropy": normalized_entropy(usage),
                "normalized_gini": normalized_gini(usage),
                "dead_experts": [i for i, x in enumerate(usage) if x <= 0.01],
                "tensor_shape": list(tensor.shape),
                "aux_loss": aux,
                "source": "router_topk_output" if self.family == "moe" else "last_routing_snapshot",
                "source_snapshot_keys": sorted(snapshot),
                "usage_scope": snapshot.get("usage_scope", "rank_local"),
                "global_usage_available": bool(snapshot.get("global_usage_available", False)),
                "dispatch_policy": snapshot.get("dispatch_policy"),
                "executed_experts": snapshot.get("executed_experts"),
            }

        return capture

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def output_tensors(value):
    """Flatten nested model outputs for a non-invasive parity check."""
    if hasattr(value, "detach"):
        return [value]
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (list, tuple)):
        return []
    return [tensor for child in value for tensor in output_tensors(child)]


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
    parser.add_argument("--config", type=Path, help="optional smoke YAML; explicit CLI arguments override it")
    parser.add_argument("--data", default="coco8.yaml", help="dataset YAML used to resolve a real validation image")
    parser.add_argument("--image", type=Path, help="optional image override; defaults to the first validation image")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/e3_routing_smoke/results" / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f"),
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or mps")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    return parser


def parse_args(argv=None):
    """Load executable YAML defaults, rejecting unused keys."""
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        import yaml

        config = yaml.safe_load(preliminary.config.read_text(encoding="utf-8"))
        allowed = {"data", "imgsz", "seed", "device", "warmup", "iterations", "families", "output", "image"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("config contains unsupported keys")
        for key in ("output", "image"):
            if key in config:
                config[key] = Path(config[key])
        if "families" in config and (
            not isinstance(config["families"], list)
            or not config["families"]
            or any(family not in PROFILES for family in config["families"])
        ):
            parser.error("invalid config families")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if len(args.families) != len(set(args.families)):
        parser.error("duplicate families")
    return args


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


def _resolve_image(data: str, override: Path | None) -> tuple[Path, dict[str, str]]:
    """Resolve one real validation image from a dataset YAML."""
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data, autodownload=False)
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


def _profile_run(family: str, args: argparse.Namespace, torch: Any, device: Any) -> dict[str, Any]:
    from tools.routing_interpreter import _load_batch, _load_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = PROFILES[family]
    model = _load_model(config, device, half=False)
    batch = _load_batch(args.image, args.imgsz, device, torch.float32)
    collector = SnapshotCollector(model, family)

    def forward():
        with torch.no_grad():
            return model(batch)

    natural = forward()
    with collector:
        observed = forward()
    left, right = output_tensors(natural), output_tensors(observed)
    if not left or len(left) != len(right):
        raise ValueError("output structure changed with hooks")
    for first, second in zip(left, right):
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("non-finite model output")
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    if set(collector.records) != set(collector.modules):
        raise ValueError("not all selected modules produced fresh records")
    layers = list(collector.records.values())
    del natural, observed, left, right

    def captured_forward():
        with collector:
            forward()

    for _ in range(args.warmup):
        forward()
        captured_forward()
    samples = {"baseline": [], "instrumented": []}
    callbacks = {"baseline": forward, "instrumented": captured_forward}
    for index in range(args.iterations):
        order = ("baseline", "instrumented") if index % 2 == 0 else ("instrumented", "baseline")
        for mode in order:
            _sync(torch, device)
            start = time.perf_counter()
            callbacks[mode]()
            _sync(torch, device)
            samples[mode].append((time.perf_counter() - start) * 1000)
    stats = {
        mode: {
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for mode, values in samples.items()
    }
    baseline, instrumented = stats["baseline"], stats["instrumented"]
    return {
        "family": family,
        "status": "passed",
        "config": str(config.relative_to(ROOT)).replace("\\", "/"),
        "initialization": "random",
        "captured_routed_layers_total": len(layers),
        "matching_layers": layers,
        "validation_errors": [],
        "output_parity": {"passed": True, "rtol": 1e-5, "atol": 1e-6},
        "hooks_removed": not collector.handles,
        "overhead": {
            "method": "alternating_paired_eval_forward",
            "warmup": args.warmup,
            "iterations": args.iterations,
            **stats,
            "median_delta_ms": instrumented["median_ms"] - baseline["median_ms"],
            "median_ratio": instrumented["median_ms"] / baseline["median_ms"],
            "scope": "temporary hooks + eval forward + CPU records + validation; excludes disk/plots",
            "note": "Diagnostic observation cost; native snapshot publication remains on in both arms. Not P1 training.",
        },
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
        right.invert_yaxis()
        right.set_xlim(0.0, 1.0)
        right.set_title("Routing balance diagnostics")
        right.legend()
    figure.suptitle("E3 routing smoke: " + " / ".join(p["family"] for p in profiles), fontsize=16)
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
        family_figure.colorbar(family_image, ax=axis, label=layers[0]["usage_semantics"])
        family_figure.tight_layout()
        family_figure.savefig(output / f"{profile['family']}_expert_usage.png", dpi=160, bbox_inches="tight")
        plt.close(family_figure)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.sha256.json":
            files[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    _write_json(output / "manifest.sha256.json", {"algorithm": "sha256", "files": files})


def run(args) -> int:
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    os.environ["MOE_SNAPSHOT_INTERVAL"] = "1"

    import torch

    from ultralytics.utils import SETTINGS

    local_datasets = ROOT.parent / "datasets"
    if (local_datasets / "coco8").is_dir():
        SETTINGS.update({"datasets_dir": str(local_datasets)})

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
        except Exception as error:  # noqa: BLE001 - record failure and continue other families
            traceback.print_exc()
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
    command_args = [
        "python",
        "scripts/e3_routing_smoke.py",
        "--data",
        args.data,
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--image",
        str(args.image),
        "--families",
        *args.families,
    ]
    command = subprocess.list2cmdline(command_args)
    snapshot = {
        "schema_version": "e3.routing_snapshot.v2",
        "purpose": "architecture and instrumentation admission smoke",
        "accuracy_claim": False,
        "dataset": args.data,
        "input": {"image": args.image.name, "imgsz": args.imgsz},
        "seed": args.seed,
        "profiles": profiles,
    }
    environment = _environment(torch, device)
    environment["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment["official_base_commit"] = "246e79cfe418cfd90f4738bace56b02245dc38f8"
    environment["moe_snapshot_interval"] = 1
    environment["openblas_num_threads"] = os.getenv("OPENBLAS_NUM_THREADS")
    environment["omp_num_threads"] = os.getenv("OMP_NUM_THREADS")
    environment["git_dirty"] = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    input_record = {
        "image": args.image.name,
        "dataset": dataset,
        "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "imgsz": args.imgsz,
        "batch_size": 1,
        "dtype": "float32",
        "preprocess": "LetterBox square with padding, BGR to RGB, CHW, divide by 255",
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
        "scope": "admission smoke and three-family prototype; not full P0/P1/P2 acceptance",
        "unsupported_families": ["moa", "molora"],
        "admission_rule": "all requested families build, expose matching routes, and emit valid normalized usage",
        "warnings": [
            "Random initialization is not evidence of model accuracy or trained expert specialization.",
            "Overhead is paired eval observation cost, not a training slowdown measurement.",
        ],
    }
    _write_json(args.output / "routing_snapshot.json", snapshot)
    _write_json(args.output / "route_stats.json", snapshot)
    _write_json(args.output / "environment.json", environment)
    _write_json(args.output / "input.json", input_record)
    _write_json(args.output / "config.resolved.json", resolved_config)
    _write_json(args.output / "summary.json", summary)
    with (args.output / "routing_snapshot.jsonl").open("w", encoding="utf-8") as stream:
        for profile in profiles:
            for layer in profile.get("matching_layers", []):
                stream.write(
                    json.dumps(
                        {
                            "schema_version": snapshot["schema_version"],
                            "seed": args.seed,
                            "step": 0,
                            "mode": "eval",
                            **layer,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
    _write_json(
        args.output / "run_config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    (args.output / "command.txt").write_text(command + "\n", encoding="utf-8")
    _save_figure(profiles, args.output)
    print(f"Result: {summary['status']} | evidence={args.output}", flush=True)
    return 0 if passed else 1


def main() -> int:
    """Keep complete logs, including startup failures, in a new run directory."""
    import contextlib

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    runtime = ROOT / "reports/e3_routing_smoke/env/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)

    args = parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.imgsz < 32 or args.imgsz % 32:
        raise SystemExit("warmup >= 0, iterations >= 1; imgsz must be a positive multiple of 32")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, text):
            self.terminal.write(text)
            self.log.write(text)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    with (args.output / "full.log").open("w", encoding="utf-8") as log:  # noqa: SIM117 - close log before hashing
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("Started:", datetime.now(timezone.utc).isoformat())
            try:
                result = run(args)
            except Exception as error:  # noqa: BLE001 - persist startup failures
                traceback.print_exc()
                _write_json(args.output / "summary.json", {"status": "failed", "error": str(error)})
                result = 1
    _manifest(args.output)
    print("Evidence:", args.output)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
