from pathlib import Path

import numpy as np
import pytest

from uav_isac.evaluation.compute_energy_calibration import (
    event_package_energy_upper_j,
)
from uav_isac.evaluation.energy_counter_window import (
    EnergyCounterSample,
    close_energy_counter_window,
    discover_linux_rapl_package_domains,
)


def _sample(
    value, timestamp, *, wrap=None, generation="g0", modulus=10.0,
    valid=True, rate=20.0,
):
    return EnergyCounterSample(
        counter_id="counter-0",
        monotonic_ns=int(timestamp),
        counter_j=(float(value) if valid else None),
        counter_resolution_j=(1.0e-6 if valid else None),
        reading_uncertainty_j=(0.001 if valid else None),
        counter_modulus_j=float(modulus),
        generation_id=generation,
        wrap_index=wrap,
        valid=valid,
        energy_rate_upper_w=rate,
    )


def test_counter_window_adds_two_reading_uncertainties_and_one_resolution():
    window = close_energy_counter_window(
        _sample(1.0, 1_000_000_000),
        _sample(1.4, 1_100_000_000),
        episode_id="e", event_id="0",
    )
    assert not window.censored
    assert window.duration_s == pytest.approx(0.1)
    assert window.observation.counter_difference_uncertainty_j == 0.002
    assert event_package_energy_upper_j(
        window.observation) == pytest.approx(0.402001)


def test_explicit_wrap_index_is_used_but_unobserved_drop_is_censored():
    wrapped = close_energy_counter_window(
        _sample(9.8, 1_000_000_000, wrap=4),
        _sample(0.2, 1_100_000_000, wrap=5),
        episode_id="e", event_id="wrap",
    )
    assert not wrapped.censored
    assert wrapped.observation.counter_wrap_count == 1
    assert event_package_energy_upper_j(
        wrapped.observation) == pytest.approx(0.402001)
    unknown = close_energy_counter_window(
        _sample(9.8, 1_000_000_000),
        _sample(0.2, 1_100_000_000),
        episode_id="e", event_id="unknown",
    )
    assert unknown.censored
    assert unknown.censored_reason == (
        "decreasing_counter_without_wrap_evidence")
    assert np.isinf(event_package_energy_upper_j(unknown.observation))


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        (_sample(1.0, 1, valid=False), _sample(1.1, 2),
         "invalid_counter_sample"),
        (_sample(1.0, 1), _sample(1.1, 2, generation="g1"),
         "counter_generation_changed"),
        (_sample(1.0, 1, wrap=2), _sample(1.1, 2, wrap=1),
         "wrap_index_regressed"),
    ],
)
def test_ambiguous_live_counter_windows_fail_closed(start, end, reason):
    result = close_energy_counter_window(
        start, end, episode_id="e", event_id=reason)
    assert result.censored
    assert result.censored_reason == reason


def test_nonpositive_window_duration_is_a_programming_error():
    with pytest.raises(ValueError, match="positive duration"):
        close_energy_counter_window(
            _sample(1.0, 2), _sample(1.1, 2),
            episode_id="e", event_id="0")


def test_no_wrap_index_requires_rate_bound_to_exclude_hidden_wraps():
    missing = close_energy_counter_window(
        _sample(1.0, 0, rate=None),
        _sample(1.1, 100_000_000, rate=None),
        episode_id="e", event_id="missing")
    assert missing.censored_reason == "missing_hidden_wrap_exclusion_bound"
    too_long = close_energy_counter_window(
        _sample(1.0, 0, rate=20.0),
        _sample(1.1, 1_000_000_000, rate=20.0),
        episode_id="e", event_id="long")
    assert too_long.censored_reason == "hidden_wrap_cannot_be_excluded"
    impossible = close_energy_counter_window(
        _sample(1.0, 0, rate=1.0),
        _sample(2.0, 100_000_000, rate=1.0),
        episode_id="e", event_id="rate")
    assert impossible.censored_reason == (
        "counter_delta_exceeds_energy_rate_bound")


def test_fake_linux_rapl_discovery_and_drop_censoring(tmp_path: Path):
    package = tmp_path / "intel-rapl-0"
    package.mkdir()
    (package / "name").write_text("package-0\n", encoding="ascii")
    (package / "max_energy_range_uj").write_text(
        "10000000\n", encoding="ascii")
    (package / "energy_uj").write_text("1000000\n", encoding="ascii")
    domains = discover_linux_rapl_package_domains(tmp_path)
    assert len(domains) == 1
    start = domains[0].read_sample(
        generation_id="boot-0", reading_uncertainty_j=0.001,
        energy_rate_upper_w=20.0, monotonic_ns=1_000_000_000)
    (package / "energy_uj").write_text("1400000\n", encoding="ascii")
    end = domains[0].read_sample(
        generation_id="boot-0", reading_uncertainty_j=0.001,
        energy_rate_upper_w=20.0, monotonic_ns=1_100_000_000)
    complete = close_energy_counter_window(
        start, end, episode_id="e", event_id="0")
    assert not complete.censored
    (package / "energy_uj").write_text("100000\n", encoding="ascii")
    dropped = domains[0].read_sample(
        generation_id="boot-0", reading_uncertainty_j=0.001,
        energy_rate_upper_w=20.0, monotonic_ns=1_200_000_000)
    censored = close_energy_counter_window(
        end, dropped, episode_id="e", event_id="1")
    assert censored.censored
    assert discover_linux_rapl_package_domains(tmp_path / "missing") == ()
