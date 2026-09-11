import numpy as np
import pytest
from uav_isac.physical.information_markov import InformationAction,solve_terminal_detection


def action(name,p,power=0.):
    return InformationAction(name,np.asarray(p,float),np.full((3,1),power),
                             np.zeros((3,1)),np.ones(3,dtype=bool))


def test_future_information_value_beats_immediate_score():
    # Explicit synthetic states: empty, buffered, delivered. Values are
    # illustrative terminal detection probabilities, not an 800m calibration.
    actions=[action('idle',np.eye(3)),
             action('acquire',[[0,1,0],[0,1,0],[0,0,1]],.6),
             action('send',[[1,0,0],[0,0,1],[0,0,1]],.4),
             action('illegal',[[0,0,1],[0,0,1],[0,0,1]],1.1)]
    r=solve_terminal_detection(actions,[.1,.1,.9],2)
    assert r['values_by_steps_remaining'][1,0]==pytest.approx(.1)
    assert r['values_by_steps_remaining'][2,0]==pytest.approx(.9)
    assert r['action_names'][r['policy_by_steps_remaining'][1,0]]=='acquire'
    assert r['action_names'][r['policy_by_steps_remaining'][0,1]]=='send'
    assert not np.any(r['policy_by_steps_remaining']==3)


def test_invalid_probability_model_rejected():
    with pytest.raises(ValueError,match='stochastic'):
        solve_terminal_detection([action('bad',np.zeros((3,3)))],[.1,.2,.8],2)
