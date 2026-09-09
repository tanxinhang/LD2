from config.params import get_default_config
from uav_isac.adapters import HoldPositionPolicy, LegacyGymEnvironmentAdapter
from uav_isac.application import EpisodeRunner
from uav_isac.domain import EpisodeSpec
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.governance.fingerprint import semantic_fingerprint


def _run(seed=451):
    config = get_default_config()
    config.scenario.T = 3
    environment = UAVISACEnv(config=config, seed=seed)
    return EpisodeRunner(
        LegacyGymEnvironmentAdapter(environment),
        HoldPositionPolicy(),
    ).run(EpisodeSpec(seed=seed, max_frames=3))


def test_semantic_fingerprint_is_repeatable_and_ignores_wall_time():
    first = _run()
    second = _run()

    assert first.frames[0].info["timing_simulator_step_wall_s"] != (
        second.frames[0].info["timing_simulator_step_wall_s"]
    )
    assert semantic_fingerprint(first) == semantic_fingerprint(second)


def test_semantic_fingerprint_changes_with_seed():
    assert semantic_fingerprint(_run(451)) != semantic_fingerprint(_run(452))

