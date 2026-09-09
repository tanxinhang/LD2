from uav_isac.adapters import HoldPositionPolicy, build_legacy_environment
from uav_isac.application import EpisodeRunner
from uav_isac.domain import EpisodeSpec
from uav_isac.governance import (
    load_characterization_baselines,
    semantic_fingerprint,
)


def test_registered_characterization_baselines_replay_exactly():
    baselines = load_characterization_baselines()
    assert {item.name for item in baselines} == {
        "default_hold_seed451_3f",
        "strict_k16q16_hold_seed451_2f",
    }
    for baseline in baselines:
        environment = build_legacy_environment(baseline.config, baseline.seed)
        result = EpisodeRunner(environment, HoldPositionPolicy()).run(
            EpisodeSpec(seed=baseline.seed, max_frames=baseline.frames)
        )
        assert semantic_fingerprint(result) == baseline.semantic_sha256

