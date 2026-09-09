from pathlib import Path

from uav_isac.adapters.legacy_process import _rewrite_outputs


def test_managed_train_forces_primary_and_auxiliary_outputs_into_run():
    run = Path("D:/research/artifacts/runs/train-abc")

    arguments = _rewrite_outputs(
        "train",
        (
            "--seed", "7",
            "--out-dir", "results/old-location",
            "--evidence-trace-output", "results/custom_trace.npz",
        ),
        run,
    )

    assert arguments[arguments.index("--out-dir") + 1] == str(run / "raw/output")
    assert arguments[arguments.index("--evidence-trace-output") + 1] == str(
        run / "raw/aux/custom_trace.npz")
    assert all("results/" not in item for item in arguments)


def test_managed_pilot_injects_canonical_output_when_omitted():
    run = Path("D:/research/artifacts/runs/pilot-abc")

    arguments = _rewrite_outputs("pilot", ("--seeds", "7,11"), run)

    assert arguments[-2:] == ("--output", str(run / "raw/result.json"))
