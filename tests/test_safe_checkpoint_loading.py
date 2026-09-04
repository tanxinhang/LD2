import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_isac.coordination.dynamic_local_search import (
    DynamicLocalSearchCoordinator,
)
from uav_isac.coordination.factor_graph_coordinator import (
    FiniteRoundFactorGraphCoordinator,
)
from uav_isac.coordination.learned_move_ranker import (
    FrozenLocalMoveRanker,
    LocalMoveRanker,
)
from uav_isac.coordination.local_move_ranker import FEATURE_NAMES
from uav_isac.utils.checkpoint_loading import (
    SafeCheckpointLoadError,
    safe_torch_load,
    validate_state_dict,
)


def _write_attack_marker(path: str):
    Path(path).write_text("executed", encoding="utf-8")
    return None


class _CustomPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_attack_marker, (str(self.marker),)


def test_safe_loader_accepts_tensors_and_primitive_metadata(tmp_path):
    checkpoint = tmp_path / "safe.pt"
    torch.save({
        "weights": {"layer.weight": torch.arange(4)},
        "metadata": {"schema": 2, "name": "safe", "enabled": True},
    }, checkpoint)

    payload = safe_torch_load(checkpoint, description="test checkpoint")

    torch.testing.assert_close(
        payload["weights"]["layer.weight"], torch.arange(4))
    assert payload["metadata"] == {
        "schema": 2, "name": "safe", "enabled": True}


def test_safe_loader_accepts_nested_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "optimizer.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": {"epoch": 1, "metrics": [0.1, 0.2]},
    }, checkpoint)

    payload = safe_torch_load(
        checkpoint,
        state_dict_keys=("model_state",),
        mapping_keys=("optimizer_state", "metadata"),
    )

    assert payload["optimizer_state"]["param_groups"][0]["lr"] == 1.0e-3
    assert payload["optimizer_state"]["state"]


def test_safe_loader_rejects_non_mapping_root(tmp_path):
    checkpoint = tmp_path / "list.pt"
    torch.save([torch.ones(1)], checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="root must be a mapping"):
        safe_torch_load(checkpoint)


def test_safe_loader_enforces_declared_schema(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    torch.save({"weights": {"layer.weight": torch.ones(1)}}, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="missing required key 'metadata'"):
        safe_torch_load(
            checkpoint,
            required_keys=("metadata",),
            state_dict_keys=("weights",),
        )


def test_state_dict_validation_rejects_non_tensor_values():
    with pytest.raises(SafeCheckpointLoadError, match="must be a tensor"):
        validate_state_dict({"layer.weight": [1.0]})


def test_safe_loader_enforces_cumulative_tensor_element_limit(tmp_path):
    checkpoint = tmp_path / "too_many_elements.pt"
    torch.save({
        "first": torch.ones(3),
        "nested": {"second": torch.ones(3)},
    }, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="exceeding the limit of 5"):
        safe_torch_load(checkpoint, max_tensor_elements=5)


def test_safe_loader_rejects_oversized_file_before_deserialization(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "oversized.pt"
    checkpoint.write_bytes(b"0123456789")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(SafeCheckpointLoadError, match="file size 10 bytes"):
        safe_torch_load(checkpoint, max_checkpoint_bytes=9)
    assert not called


@pytest.mark.parametrize("bad_value", [
    torch.tensor([float("nan")], dtype=torch.float32),
    torch.tensor([complex(1.0, float("inf"))], dtype=torch.complex64),
])
def test_safe_loader_rejects_nonfinite_tensor_values(tmp_path, bad_value):
    checkpoint = tmp_path / "nonfinite_tensor.pt"
    torch.save({"state": {"weight": bad_value}}, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="non-finite tensor value"):
        safe_torch_load(checkpoint, state_dict_keys=("state",))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_safe_loader_rejects_meta_tensor_for_every_dtype(tmp_path, dtype):
    checkpoint = tmp_path / f"meta_{dtype}.pt"
    torch.save({
        "state": {"weight": torch.empty(2, device="meta", dtype=dtype)},
    }, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="meta device"):
        safe_torch_load(checkpoint, state_dict_keys=("state",))


def test_safe_loader_rejects_nonfinite_numeric_metadata(tmp_path):
    checkpoint = tmp_path / "nonfinite_metadata.pt"
    torch.save({"metadata": {"score": float("nan")}}, checkpoint)

    with pytest.raises(
        SafeCheckpointLoadError, match="non-finite numeric metadata",
    ):
        safe_torch_load(checkpoint, mapping_keys=("metadata",))


def test_safe_loader_narrowly_supports_legacy_numpy_float32(tmp_path):
    checkpoint = tmp_path / "legacy_numpy.pt"
    expected = np.asarray([1.0, 2.0], dtype=np.float32)
    torch.save({"normalization": expected}, checkpoint)

    payload = safe_torch_load(
        checkpoint, allow_legacy_numpy_float32=True)

    np.testing.assert_array_equal(payload["normalization"], expected)


def test_safe_loader_rejects_nonfinite_legacy_numpy_array(tmp_path):
    checkpoint = tmp_path / "legacy_numpy_nonfinite.pt"
    torch.save({
        "normalization": np.asarray([1.0, np.inf], dtype=np.float32),
    }, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="non-finite array value"):
        safe_torch_load(
            checkpoint, allow_legacy_numpy_float32=True)


def test_safe_loader_rejects_custom_pickle_without_execution(tmp_path):
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "attack_executed.txt"
    torch.save({"payload": _CustomPayload(marker)}, checkpoint)

    with pytest.raises(SafeCheckpointLoadError) as caught:
        safe_torch_load(
            checkpoint,
            description="malicious test checkpoint",
            allow_legacy_numpy_float32=True,
        )

    assert not marker.exists()
    message = str(caught.value)
    assert "No unsafe pickle fallback was attempted" in message
    assert "Re-export" in message
    assert "add_safe_globals" not in message


def test_safe_loader_rejects_vulnerable_torch_before_deserialization(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "never-opened.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(torch, "__version__", "2.9.1")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(SafeCheckpointLoadError, match=r"upgrade to PyTorch >=2\.10"):
        safe_torch_load(checkpoint)
    assert not called


def test_local_move_ranker_loads_restricted_legacy_artifact(tmp_path):
    width = len(FEATURE_NAMES)
    model = LocalMoveRanker(
        np.zeros(width, dtype=np.float32),
        np.ones(width, dtype=np.float32),
        hidden_dim=8,
    )
    checkpoint = tmp_path / "ranker.pt"
    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": np.zeros(width, dtype=np.float32),
        "feature_std": np.ones(width, dtype=np.float32),
        "hidden_dim": 8,
        "alpha": 0.25,
    }, checkpoint)

    frozen = FrozenLocalMoveRanker(checkpoint)

    prediction = frozen.predict(np.zeros((2, width), dtype=np.float32))
    assert prediction.shape == (2,)
    assert np.all(np.isfinite(prediction))


def test_factor_graph_initializer_loads_restricted_artifact(tmp_path):
    model = FiniteRoundFactorGraphCoordinator(
        edge_feature_dim=4, hidden_dim=8, rounds=2)
    checkpoint = tmp_path / "factor_graph.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": 8,
        "rounds": 2,
        "edge_feature_dim": 4,
        "coupling_strength": 1.0,
        "coupling_rounds": 0,
        "use_global_context": False,
    }, checkpoint)

    coordinator = DynamicLocalSearchCoordinator(
        "previous",
        cold_initializer="factor_graph",
        factor_graph_checkpoint=checkpoint,
    )

    assert coordinator.factor_graph is not None
    assert not coordinator.factor_graph.training


def test_factor_graph_initializer_rejects_incomplete_schema(tmp_path):
    checkpoint = tmp_path / "factor_graph_missing_rounds.pt"
    torch.save({
        "state_dict": {"weight": torch.ones(1)},
        "hidden_dim": 8,
    }, checkpoint)

    with pytest.raises(SafeCheckpointLoadError, match="missing required key 'rounds'"):
        DynamicLocalSearchCoordinator(
            "previous",
            cold_initializer="factor_graph",
            factor_graph_checkpoint=checkpoint,
        )


@pytest.mark.parametrize("relative_path", [
    "scripts/run_mappo.py",
    "scripts/test_ppo_ratio_fix.py",
    "uav_isac/agents/frozen_structure_student.py",
    "uav_isac/agents/equivariant_movement_plan.py",
    "uav_isac/coordination/dynamic_local_search.py",
    "uav_isac/coordination/learned_move_ranker.py",
])
def test_deployment_checkpoint_paths_do_not_call_torch_load_directly(
    relative_path,
):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    direct_loads = [
        node for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        )
    ]
    assert direct_loads == []


def test_repository_has_no_checkpoint_bypass_outside_safe_loader():
    root = Path(__file__).resolve().parents[1]
    violations = []
    safe_loader = root / "uav_isac" / "utils" / "checkpoint_loading.py"
    for source_root in ("scripts", "tools", "uav_isac", "tests"):
        for path in (root / source_root).rglob("*.py"):
            if path == safe_loader:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    violations.append(
                        f"{path.relative_to(root).as_posix()}:{node.lineno}")
    assert violations == []
