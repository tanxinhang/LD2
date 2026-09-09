import json
from pathlib import Path
import subprocess
import sys


def test_strict_stack_overrides_non_power_actor_fields():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "audit_actor_head_responsibility.py"),
            "--config",
            str(root / "config" / "exp_strict_distributed_no_truth_pilot.yaml"),
            "--seed",
            "101",
            "--frames",
            "4",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["formal_runner_has_learned_actor"] is False
    probes = result["counterfactual_field_interventions"]
    for name in (
        "movement",
        "role",
        "message_content",
        "rate",
        "sensing_weights",
    ):
        assert probes[name]["physical_changed"] is False
    assert probes["comm_power_fraction"]["physical_changed"] is True
    assert probes["comm_power_fraction"]["structure_changed"] is False
    assert probes["comm_power_fraction"]["max_abs_delta"]["detection"] == 0.0
