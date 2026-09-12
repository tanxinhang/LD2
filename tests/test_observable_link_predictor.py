import numpy as np
from uav_isac.physical.observable_link_predictor import fit_delivery_markov,predict_delivery_paths


def test_order_changes_prediction_and_paths_are_nested():
    transition=fit_delivery_markov(np.array([[False,False,True,True]]*20))
    a=predict_delivery_paths(np.array([False,True]),transition,9,64,42)
    b=predict_delivery_paths(np.array([True,False]),transition,9,64,42)
    assert not np.array_equal(a,b)
    assert np.array_equal(a,predict_delivery_paths(np.array([False,True]),transition,9,256,42)[:64])
    assert np.allclose(transition.sum(axis=1),1)


def test_frozen_threshold_is_not_replaced_by_evaluation_calibration():
    from tools.audit_unified_receiver_transport import audit
    result=audit(samples=200,window=6,report_gate=True,frozen_gate_threshold=7.)
    row=next(x for x in result['rows'] if x['method']=='prefix_gated_mixture')
    assert row['threshold']==7.
