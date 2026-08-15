from types import SimpleNamespace

import numpy as np
import torch

from tools.train_multiscale_structure_student import (
    _balanced_normalization,
    _initialize_from_preservation_student,
    _seed_grouped_split,
    _selection_key,
)
from uav_isac.agents.frozen_structure_student import (
    FactorizedStructureStudent,
    FrozenStructureStudent,
    save_frozen_structure_student,
)


def test_seed_grouped_split_is_disjoint_and_resolved_only():
    data = {
        "seed": np.repeat(np.arange(10), 3),
        "p0_resolved": np.tile([1, 0, 1], 10),
    }
    split_seed, split_index = _seed_grouped_split(data)
    assert set(split_seed) == {"fit", "validation", "test"}
    assert all(np.all(data["p0_resolved"][value]) for value in split_index.values())
    assert not (set(split_seed["fit"]) & set(split_seed["validation"]))
    assert not (set(split_seed["fit"]) & set(split_seed["test"]))
    assert not (set(split_seed["validation"]) & set(split_seed["test"]))

    filtered_seed, filtered_index = _seed_grouped_split(
        data, np.asarray([2, 7]))
    assert 2 not in np.concatenate(list(filtered_seed.values()))
    assert 7 not in np.concatenate(list(filtered_seed.values()))
    assert all(
        not np.any(np.isin(data["seed"][value], [2, 7]))
        for value in filtered_index.values())


def test_balanced_normalization_gives_each_cardinality_equal_weight():
    small = SimpleNamespace(features={
        "fit": np.zeros((100, 2, 2, 1), dtype=np.float32)})
    large = SimpleNamespace(features={
        "fit": np.full((1, 4, 4, 1), 10.0, dtype=np.float32)})
    mean, scale = _balanced_normalization([small, large])
    np.testing.assert_allclose(mean, [5.0])
    np.testing.assert_allclose(scale, [5.0])


def test_normalization_migration_preserves_raw_student_function(tmp_path):
    torch.manual_seed(4)
    original = FactorizedStructureStudent(
        input_dim=3, hidden_dim=5, endpoint_dim=2)
    old_mean = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    old_scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    checkpoint = tmp_path / "student.pt"
    save_frozen_structure_student(
        checkpoint,
        original,
        old_mean,
        old_scale,
        num_uavs=2,
        num_targets=2,
        rate_scale=1.0,
    )
    frozen = FrozenStructureStudent(checkpoint)
    migrated = FactorizedStructureStudent(
        input_dim=3, hidden_dim=5, endpoint_dim=2)
    new_mean = np.asarray([-3.0, 4.0, 1.5], dtype=np.float32)
    new_scale = np.asarray([5.0, 6.0, 7.0], dtype=np.float32)
    _initialize_from_preservation_student(
        migrated, frozen, new_mean, new_scale)
    raw = torch.randn(3, 2, 2, 3)
    with torch.inference_mode():
        old_normalized = (
            (raw - torch.as_tensor(old_mean)) / torch.as_tensor(old_scale))
        new_normalized = (
            (raw - torch.as_tensor(new_mean)) / torch.as_tensor(new_scale))
        old_value = frozen.model(old_normalized)
        new_value = migrated(new_normalized)
        old_endpoint = frozen.model.encode_endpoints(old_normalized)
        new_endpoint = migrated.encode_endpoints(new_normalized)
    torch.testing.assert_close(new_value, old_value, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        new_endpoint[0], old_endpoint[0], atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        new_endpoint[1], old_endpoint[1], atol=1.0e-6, rtol=1.0e-6)


def test_checkpoint_selection_protects_weakest_scale_first():
    def report(feasible_a, worst_a, feasible_b, worst_b):
        return {
            "4/4": {"projection": {"realized_pd": {
                "qos_feasible_rate": feasible_a, "worst": worst_a}}},
            "6/6": {"projection": {"realized_pd": {
                "qos_feasible_rate": feasible_b, "worst": worst_b}}},
        }

    balanced = _selection_key(report(0.8, 0.7, 0.7, 0.6))
    averaged_but_unsafe = _selection_key(report(1.0, 0.9, 0.6, 0.7))
    assert balanced > averaged_but_unsafe
