import json

import numpy as np

from tools.export_predictive_teacher_dataset import (
    SCHEMA_VERSION,
    export_predictive_teacher_dataset,
    export_predictive_teacher_payload,
)


def test_export_predictive_teacher_dataset_pads_edges(tmp_path):
    def snapshot(frame, selected):
        return {
            "frame": frame,
            "hold_active": bool(frame),
            "endpoint_position_views": np.zeros((2, 2, 2, 2)).tolist(),
            "endpoint_velocity_views": np.zeros((2, 2, 2, 2)).tolist(),
            "target_position_views": np.zeros((2, 2, 3)).tolist(),
            "target_velocity_views": np.zeros((2, 2, 3)).tolist(),
            "target_position_uncertainty_views": np.zeros((2, 2)).tolist(),
            "target_velocity_uncertainty_views": np.zeros((2, 2)).tolist(),
            "visible_views": np.ones((2, 2, 2), dtype=bool).tolist(),
            "public_gain_views": np.ones((2, 2, 2)).tolist(),
            "certificate_gain_views": np.full((2, 2, 2), 0.5).tolist(),
            "certificate_gain_upper_views": np.full((2, 2, 2), 1.5).tolist(),
            "local_power_cache_w": np.full((2, 2, 2), 0.1).tolist(),
            "local_prices": np.full((2, 2), 0.25).tolist(),
            "local_cache_valid": [True, True],
            "selected": selected,
        }

    source = tmp_path / "trace.json"
    output = tmp_path / "teacher.npz"
    source.write_text(json.dumps({
        "episodes": [{
            "seed": 7,
            "trace": {
                "acceleration_golden": [
                    snapshot(0, [[0, 1, 0]]),
                    snapshot(1, [[0, 1, 0], [1, 0, 1]]),
                ],
                "detection": [[0.6, 0.7], [0.7, 0.8]],
                "detection_deflection": [[10.0, 11.0], [12.0, 13.0]],
                "sensing_power_w": [
                    np.full((2, 2), 0.1).tolist(),
                    np.full((2, 2), 0.1).tolist(),
                ],
                "certificate_safe_gain_per_watt": [
                    np.ones((2, 2)).tolist(),
                    np.ones((2, 2)).tolist(),
                ],
                "bits": [100.0, 80.0],
                "delivery": [1.0, 0.9],
            },
        }],
    }), encoding="utf-8")

    summary = export_predictive_teacher_dataset(source, output)

    assert summary["frames"] == 2
    assert summary["max_edges"] == 2
    with np.load(output) as dataset:
        assert str(dataset["schema_version"]) == SCHEMA_VERSION
        assert dataset["endpoint_position"].shape == (2, 2, 2, 2, 2)
        np.testing.assert_array_equal(dataset["edge_mask"], [
            [True, False], [True, True],
        ])
        np.testing.assert_array_equal(dataset["edge_index"][0, 1], [-1, -1, -1])
        assert dataset["executed_power"].shape == (2, 2, 2)
        assert dataset["executed_nominal_gain"].shape == (2, 2, 2)
        assert dataset["actual_deflection"].shape == (2, 2)
        np.testing.assert_allclose(dataset["protocol_bits"], [100.0, 80.0])

    second = tmp_path / "teacher_from_memory.npz"
    payload = json.loads(source.read_text(encoding="utf-8"))
    memory_summary = export_predictive_teacher_payload(payload, second)
    assert memory_summary["frames"] == summary["frames"]
    with np.load(second) as memory_dataset, np.load(output) as file_dataset:
        assert set(memory_dataset.files) == set(file_dataset.files)
        for key in memory_dataset.files:
            np.testing.assert_array_equal(memory_dataset[key], file_dataset[key])
