"""Unit tests for the dependency-light E3 smoke helpers."""

import pytest

from scripts import e3_routing_smoke as MODULE


@pytest.mark.parametrize(
    ("values", "entropy", "gini"),
    [
        ([0.25, 0.25, 0.25, 0.25], 1.0, 0.0),
        ([1.0, 0.0, 0.0, 0.0], 0.0, 1.0),
    ],
)
def test_balance_metrics_have_expected_limits(values, entropy, gini):
    assert MODULE.normalized_entropy(values) == pytest.approx(entropy)
    assert MODULE.normalized_gini(values) == pytest.approx(gini)


@pytest.mark.parametrize("values", [[float("nan"), 1], [-1, 2], [0, 0], [0.5], [0.3, 0.3]])
def test_invalid_source_is_not_repaired(values):
    with pytest.raises(ValueError):
        MODULE.validate_vector(values, 2, "test")


def test_nearest_rank_percentile():
    assert MODULE.percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert MODULE.percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_config_is_used_and_cli_overrides(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text("families: [mot]\nseed: 7\nimgsz: 64\ndevice: cpu\n", encoding="utf-8")
    args = MODULE.parse_args(["--config", str(config), "--seed", "9"])
    assert (args.families, args.seed, args.imgsz, args.device) == (["mot"], 9, 64, "cpu")
    config.write_text("ignored_typo: 3\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--config", str(config)])


@pytest.mark.parametrize("failure", ["none", "invalid", "stale", "forward"])
def test_hook_cleanup_and_freshness(failure):
    import torch

    class Producer(torch.nn.Module):
        _routing_aux_kind = "mot"

        def __init__(self):
            super().__init__()
            self.last_routing_snapshot = {}

        def forward(self, x):
            if failure == "forward":
                raise ValueError("forward failed")
            if failure != "stale":
                self.last_routing_snapshot = {
                    "num_experts": 2,
                    "top_k": 1,
                    "expert_usage": [float("nan"), 1] if failure == "invalid" else [0.5, 0.5],
                }
            return x + 1

    producer = Producer().eval()
    model = torch.nn.Sequential(producer)
    collector = MODULE.SnapshotCollector(model, "mot")
    if failure == "none":
        with collector:
            output = model(torch.zeros(1, 2))
        assert torch.equal(output, torch.ones(1, 2))
        assert collector.records["0"]["expert_usage"] == [0.5, 0.5]
    else:
        with pytest.raises(ValueError), collector:
            model(torch.zeros(1, 2))
    assert not collector.handles and not producer._forward_hooks


def test_moe_adapter_uses_dispatch_indices_not_internal_logits():
    import torch

    class Router(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.8, 0.2]]), torch.tensor([[1, 3]]), {}

    class OptimizedMOEImproved(torch.nn.Module):
        _routing_aux_kind = "moe"
        num_experts = 4

        def __init__(self):
            super().__init__()
            self.routing = Router()
            self.last_routing_snapshot = {}

        def forward(self, x):
            self.routing(x)
            return x

    producer = OptimizedMOEImproved().eval()
    collector = MODULE.SnapshotCollector(torch.nn.Sequential(producer), "moe")
    with collector:
        producer(torch.zeros(1, 4))
    record = collector.records["0"]
    assert record["expert_usage"] == [0, 0.5, 0, 0.5]
    assert record["mean_router_probs"] == pytest.approx([0, 0.8, 0, 0.2])
    assert record["aux_loss"]["observed"] is None
    assert not producer._forward_hooks and not producer.routing._forward_hooks


def test_manifest_covers_logs_and_last_written_files(tmp_path):
    import hashlib
    import json

    (tmp_path / "full.log").write_text("done", encoding="utf-8")
    (tmp_path / "command.txt").write_text("command", encoding="utf-8")
    MODULE._manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.sha256.json").read_text())
    assert set(manifest["files"]) == {"full.log", "command.txt"}
    for name, item in manifest["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == item["sha256"]
