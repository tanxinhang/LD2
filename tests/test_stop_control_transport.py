import numpy as np
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.stop_control_transport import ReservedSlotTransport


def radio():
    return InterUAVCommunicationModel(rate_bits_per_dim=[0,8],header_bits=64,
        bandwidth_hz=1e6,deadline_s=.005,processing_delay_s=.0002,
        snr_threshold_db=0.,antenna_gain_dbi=16.,carrier_hz=28e9,tx_power_w=.1,
        kT=4e-21,noise_figure_db=4.,dt=.006024,finite_blocklength_enabled=True,
        snr_shadowing_std_db=4.,snr_shadowing_correlation=.7)


def test_silent_control_has_zero_cost_and_same_future_channel():
    positions=np.array([[-800.,0.,20.],[800.,0.,20.],[0.,800.,20.]])
    for seed in range(30):
        active=ReservedSlotTransport(radio(),seed)
        silent=ReservedSlotTransport(radio(),seed)
        for _ in range(4):
            active.send(positions,2); silent.send(positions,2)
        active.send(positions,1)
        stop,stats=silent.stop_control(positions,False)
        assert not stop and stats.total_bits==0
        for _ in range(9):
            a,sa=active.send(positions,2); b,sb=silent.send(positions,2)
            assert [(p.sender,p.receiver) for p in a]==[(p.sender,p.receiver) for p in b]
            assert sa.total_bits==sb.total_bits
            assert np.array_equal(active.radio.get_channel_state()['snr_shadowing_db'],
                                  silent.radio.get_channel_state()['snr_shadowing_db'])


def test_requested_stop_charges_packet_even_if_delivery_fails():
    positions=np.array([[0.,0.,20.],[800.,0.,20.],[-800.,0.,20.]])
    model=radio()
    model.snr_threshold_db=1e6  # deterministic rejection, no extreme geometry
    transport=ReservedSlotTransport(model,123)
    stopped,stats=transport.stop_control(positions,True)
    assert not stopped and stats.total_bits==128
