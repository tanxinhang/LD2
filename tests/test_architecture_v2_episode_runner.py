import numpy as np

from config.params import get_default_config
from uav_isac.adapters import HoldPositionPolicy, LegacyGymEnvironmentAdapter
from uav_isac.application import EpisodeRunner
from uav_isac.domain import EpisodeSpec
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.governance.fingerprint import is_runtime_telemetry_key


def _assert_nested_equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            if is_runtime_telemetry_key(key):
                continue
            _assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, (float, np.floating)) and np.isnan(expected):
        assert np.isnan(actual), path
    else:
        assert actual == expected, path


def test_episode_runner_is_behaviorally_identical_to_direct_legacy_calls():
    config = get_default_config()
    config.scenario.T = 3
    seed = 451

    direct = UAVISACEnv(config=config, seed=seed)
    direct_obs, direct_initial_info = direct.reset(seed=seed)
    policy = HoldPositionPolicy()
    direct_frames = []
    for frame in range(3):
        direct_frames.append(direct.step(policy.actions(direct_obs, frame)))
        direct_obs = direct_frames[-1][0]

    migrated = UAVISACEnv(config=config, seed=seed)
    result = EpisodeRunner(
        LegacyGymEnvironmentAdapter(migrated),
        policy,
    ).run(EpisodeSpec(seed=seed, max_frames=3))

    _assert_nested_equal(result.initial_info, direct_initial_info)
    assert len(result.frames) == len(direct_frames)
    for frame, expected in zip(result.frames, direct_frames):
        actual = (
            frame.observations,
            frame.rewards,
            frame.terminated,
            frame.truncated,
            frame.info,
        )
        _assert_nested_equal(actual, expected)


def test_episode_spec_rejects_empty_runs():
    try:
        EpisodeSpec(seed=1, max_frames=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero-frame episode was accepted")
