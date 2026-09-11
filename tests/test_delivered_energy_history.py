import pytest
from uav_isac.physical.delivered_energy_history import DeliveredEnergyHistory


def test_only_received_unique_unexpired_values_enter_history():
    history=DeliveredEnergyHistory(2);history.advance(0)
    assert not history.accept((0,2,0),[100.],available=False)
    assert history.accept((0,1,0),[2.],available=True)
    assert not history.accept((0,1,0),[2.],available=True)
    assert history.total()[0]==1
    with pytest.raises(ValueError,match='conflicting'):
        history.accept((0,1,0),[3.],available=True)
    with pytest.raises(ValueError,match='future'):
        history.accept((1,1,0),[2.],available=True)
    history.advance(2)
    assert history.total()[0]==0
    assert not history.accept((0,1,0),[2.],available=True)
