import numpy as np
import pytest
from uav_isac.physical.history_power_choice import PowerAction, choose_history_conditioned_power


def choose(known=(), **kwargs):
    # Prospective sensing is strongly redundant with acknowledged history.
    return choose_history_conditioned_power(
        [PowerAction('sense',1.,0.,1,1.),PowerAction('report',0.,.4,2,.9),
         PowerAction('over_budget',1.,.1,2,1.)],
        np.array([2.,2.,1.5]),np.array([[1.,.99,0.],[.99,1.,0.],[0.,0.,1.]]),
        evidence_ids=('old','new','buffered'),observed_frames=(8,10,9),
        known_available_ids=known,current_frame=10,window_frames=kwargs.get('window',4))


def test_history_changes_choice_and_repeated_delivery_is_not_new_information():
    assert choose()['action']=='sense'
    assert choose(('old',))['action']=='report'
    assert choose(('old','buffered'))['action']=='sense'
    assert choose(('old','new','buffered'))['action']=='idle'
    assert choose(('old',),window=1)['action']=='sense'
    for known in ((),('old',),('old','buffered')):
        result=choose(known)
        assert result['sensing_w']+result['communication_w']<=1.


def test_unknown_remote_history_is_rejected():
    with pytest.raises(ValueError,match='model descriptor'):
        choose(('unreceived_remote',))
