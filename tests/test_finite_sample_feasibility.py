import pytest

from uav_isac.evaluation.finite_sample_feasibility import (
    clopper_pearson_one_sided_upper,
    conformal_sample_requirement,
    minimum_split_conformal_calibration_episodes,
    minimum_zero_failure_validation_episodes,
    split_conformal_rank,
    zero_failure_validation_requirement,
)


def test_five_percent_certificate_needs_19_calibration_and_59_validation_episodes():
    assert minimum_split_conformal_calibration_episodes(0.05) == 19
    assert split_conformal_rank(18, 0.05) == 19
    assert split_conformal_rank(19, 0.05) == 19
    assert minimum_zero_failure_validation_episodes(0.05) == 59
    assert clopper_pearson_one_sided_upper(0, 58) > 0.05
    assert clopper_pearson_one_sided_upper(0, 59) <= 0.05


def test_ten_percent_joint_link_validation_needs_29_zero_failure_episodes():
    assert minimum_split_conformal_calibration_episodes(0.10) == 9
    assert minimum_zero_failure_validation_episodes(0.10) == 29
    requirement = zero_failure_validation_requirement(
        "link_joint",
        miscoverage=0.10,
        validation_episode_count=2,
    )
    assert requirement.minimum_episode_count == 29
    assert not requirement.attainable_in_best_case


def test_requirement_reports_unattainable_conformal_rank_and_validates_inputs():
    requirement = conformal_sample_requirement(
        "runtime",
        miscoverage=0.05,
        calibration_episode_count=18,
    )
    assert requirement.rank_one_based == 19
    assert not requirement.attainable_in_best_case
    with pytest.raises(ValueError, match="miscoverage"):
        minimum_zero_failure_validation_episodes(0.0)
