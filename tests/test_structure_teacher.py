import numpy as np

from uav_isac.evaluation.structure_teacher import (
    deflection_entry_tensors,
    structure_teacher_labels,
)
from uav_isac.utils.types import DeflectionEntry


def test_structure_teacher_labels_keep_directed_roles_and_owner() -> None:
    labels = structure_teacher_labels(
        [(0, 2, 0), (1, 2, 0), (1, 3, 1)],
        num_uavs=4,
        num_targets=2,
    )
    assert labels["pair"].shape == (4, 4, 2)
    assert labels["pair"][0, 2, 0] == 1
    assert labels["tx_target"].tolist() == [
        [1, 0], [1, 1], [0, 0], [0, 0]]
    assert labels["rx_target"].tolist() == [
        [0, 0], [0, 0], [1, 0], [0, 1]]
    assert labels["endpoint_target"].sum() == 5
    assert labels["receiver_owner"].tolist() == [2, 3]
    assert labels["role"].tolist() == [0, 0, 1, 1]


def test_structure_teacher_marks_multiple_receiver_owners_invalid() -> None:
    labels = structure_teacher_labels(
        [(0, 2, 0), (1, 3, 0)],
        num_uavs=4,
        num_targets=1,
    )
    assert labels["receiver_owner"].tolist() == [-2]


def test_deflection_entry_tensors_preserve_privileged_fields() -> None:
    entry = DeflectionEntry(
        i=0,
        j=1,
        q=0,
        tau=1.0e-6,
        nu=2.0,
        alpha=0.25,
        d_raw=3.0,
        g_dd=0.75,
        chi_rep=0.5,
        d_eff=1.5,
    )
    tensors = deflection_entry_tensors(
        [entry], num_uavs=2, num_targets=1)
    assert tensors["candidate"].dtype == np.uint8
    assert tensors["candidate"][0, 1, 0] == 1
    assert tensors["candidate"][1, 0, 0] == 0
    assert np.isclose(tensors["d_eff"][0, 1, 0], 1.5)
    assert np.isclose(tensors["d_raw"][0, 1, 0], 3.0)
    assert np.isclose(tensors["alpha"][0, 1, 0], 0.25)
    assert np.isclose(tensors["g_dd"][0, 1, 0], 0.75)
    assert np.isclose(tensors["chi_rep"][0, 1, 0], 0.5)
