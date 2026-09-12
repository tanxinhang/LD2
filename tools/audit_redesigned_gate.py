"""Large-H0 design, independent calibration, frozen held-out validation."""
import json
from tools.audit_unified_receiver_transport import audit


def run():
    kwargs=dict(report_gate=True,mixture_rollout=True,rollouts=64,
        link_transition=[[.5770750988142292,.42292490118577075],
                         [.1490857946554149,.8509142053445851]])
    threshold=4.1
    for step in range(3):
        r=audit(samples=100000,trial_seed=840001,rollout_threshold=threshold,null_only=True,**kwargs)
        new=r['thresholds']['prefix_gated_mixture']
        print(json.dumps(dict(stage='design',step=step,forecast=threshold,calibrated=new)),flush=True)
        if abs(new-threshold)<1e-10: break
        threshold=new
    # Calibration does not update the policy: freeze forecast threshold first.
    calibration=audit(samples=100000,trial_seed=850001,rollout_threshold=threshold,null_only=True,**kwargs)
    detection_threshold=calibration['thresholds']['prefix_gated_mixture']
    print(json.dumps(dict(stage='independent_calibration',forecast=threshold,
        detection_threshold=detection_threshold,tail_upper=calibration['tail_probability_upper95'])),flush=True)
    # Two distinct frozen thresholds are explicit: no false self-consistency
    # claim after independent calibration. Validation evaluates this exact pair.
    r=audit(samples=100000,trial_seed=860001,rollout_threshold=threshold,
        frozen_gate_threshold=detection_threshold,**kwargs)
    print(json.dumps(dict(stage='validation',rows=[x for x in r['rows'] if x['method'] in
        ('prefix_gated_mixture','delivered_uncertainty_mixture')],
        comparison=r['gate_comparison'],cost=r['gate_metrics'])),flush=True)


if __name__=='__main__': run()
