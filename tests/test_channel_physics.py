import numpy as np
import pytest

from uav_isac.physical.channel import generate_rician_channel


@pytest.mark.parametrize("k_db", [-10.0, 0.0, 6.0, 20.0])
def test_rician_channel_preserves_configured_average_path_gain(k_db):
    path_gain = 0.25
    rng = np.random.default_rng(103)
    samples = np.fromiter(
        (
            abs(generate_rician_channel(path_gain, k_db, rng)) ** 2
            for _ in range(50_000)
        ),
        dtype=np.float64,
        count=50_000,
    )
    # E|h|^2 = path_gain for every K.  The tolerance is much wider than the
    # seeded Monte Carlo standard error but narrow enough to catch the former
    # missing sqrt(1/(K+1)) factor (about +80% at K=6 dB).
    assert float(np.mean(samples)) == pytest.approx(path_gain, rel=0.02)
