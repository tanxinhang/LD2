from __future__ import annotations

import numpy as np

from tools.audit_oracle_ladder import (
    ladder_deltas,
    oracleize_student_observation,
)
from uav_isac.environment.observation_slices import ObservationSlices


def test_oracleize_student_observation_replaces_only_state_fields():
    slices = ObservationSlices.from_config(
        K=2,
        Q=1,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=False,
    )
    obs = np.full((1, 2, slices.total_dim), 0.25, dtype=np.float32)
    targets = np.asarray([[[50.0, 80.0, 5.0, -5.0]]])
    uavs = np.asarray([[[10.0, 20.0, 10.0], [20.0, 40.0, 10.0]]])
    actual = oracleize_student_observation(
        obs,
        targets,
        uavs,
        slices,
        area_size=(100.0, 100.0),
        velocity_scale=10.0,
    )

    belief = slices.extract_beliefs(actual)
    np.testing.assert_allclose(
        belief[..., :4],
        np.asarray([[[[0.5, 0.8, 0.5, -0.5]],
                     [[0.5, 0.8, 0.5, -0.5]]]]),
    )
    np.testing.assert_allclose(belief[..., 4:], 0.0)
    geometry = slices.extract_geometry(actual)
    np.testing.assert_allclose(geometry[0, 0, 0, :2], [0.4, 0.6])
    np.testing.assert_allclose(geometry[0, 1, 0, :2], [0.3, 0.4])
    # P_D, communication, self state and unrelated fields are preserved.
    changed = np.zeros_like(obs, dtype=bool)
    changed[..., slices.belief_start:slices.geom_start] = True
    changed[..., slices.geom_start:slices.physics_start] = True
    np.testing.assert_array_equal(actual[~changed], obs[~changed])


def test_ladder_deltas_are_incremental_not_against_baseline():
    def row(worst: float) -> dict[str, float]:
        return {
            "steady": worst + 0.3,
            "weak3": worst + 0.1,
            "worst": worst,
            "cvar": worst - 0.1,
            "qos_feasible": worst,
        }

    summary = {
        "baseline": row(0.1),
        "oracle_state": row(0.1),
        "oracle_candidate": row(0.2),
        "oracle_rank": row(0.5),
        "oracle_projection": row(0.5),
    }
    delta = ladder_deltas(summary)
    assert delta["baseline_to_oracle_state"]["worst"] == 0.0
    assert np.isclose(
        delta["oracle_state_to_oracle_candidate"]["worst"], 0.1)
    assert np.isclose(
        delta["oracle_candidate_to_oracle_rank"]["worst"], 0.3)
    assert delta["oracle_rank_to_oracle_projection"]["worst"] == 0.0
