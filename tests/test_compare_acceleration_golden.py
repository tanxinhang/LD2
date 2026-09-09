from copy import deepcopy

from tools.compare_acceleration_golden import compare_results


def _result():
    frame = {
        "frame": 0,
        "hold_active": False,
        "received_last_seen": [[[0]]],
        "visible_views": [[[True]]],
        "mutual": [[0, 1, 0]],
        "stable": [[0, 1, 0]],
        "selected": [[0, 1, 0]],
        "public_full_views": [True],
        "consensus_streak": [[[1]]],
        "local_cache_valid": [True],
        "public_gain_views": [[[2.0]]],
        "certificate_gain_views": [[[1.5]]],
        "certificate_gain_upper_views": [[[2.5]]],
        "executed_power_w": [[0.1]],
        "local_power_cache_w": [[[0.1]]],
        "local_prices": [[1.0]],
        "local_plans": [{
            "selected": [[0, 1, 0]],
            "proxy_target_value": [2.0],
            "proxy_scores": [[[[0, 1, 0]], 2.0]],
        }],
    }
    return {
        "episodes": [{
            "seed": 7,
            "trace": {
                "detection": [[0.8]],
                "detection_deflection": [[4.0]],
                "sensing_power_w": [[[0.1]]],
                "certificate_safe_gain_per_watt": [[[1.5]]],
                "acceleration_golden": [frame],
            },
        }],
    }


def test_golden_comparator_accepts_exact_copy():
    reference = _result()
    report = compare_results(reference, deepcopy(reference))
    assert report["passed"]
    assert report["exact_mismatch_count"] == 0


def test_golden_comparator_rejects_structure_or_numeric_drift():
    reference = _result()
    candidate = deepcopy(reference)
    frame = candidate["episodes"][0]["trace"]["acceleration_golden"][0]
    frame["selected"] = []
    frame["public_gain_views"][0][0][0] += 1.0e-4
    report = compare_results(reference, candidate, atol=1.0e-8, rtol=1.0e-8)
    assert not report["passed"]
    assert report["exact_mismatch_count"] == 1
    assert "public_gain_views" in report["numerical_failures"]
