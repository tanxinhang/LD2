"""Prespecified adjacent-channel diagnostic; conditional per-trace calibration."""
import json
from tools.audit_unified_receiver_transport import audit


if __name__=='__main__':
    for index,link_seed in enumerate((650001,650002,650003)):
        options=dict(report_gate=True,stop_only_transport=True,mixture_rollout=True,
            rollouts=64,rollout_threshold=4.416818344356382,link_seed=link_seed,
            link_transition=[[.5770750988142292,.42292490118577075],
                             [.1490857946554149,.8509142053445851]])
        seed=890001+index*10000
        calibration=audit(samples=50000,trial_seed=seed,null_only=True,**options)
        threshold=calibration['thresholds']['prefix_gated_mixture']
        result=audit(samples=50000,trial_seed=seed+1000,frozen_gate_threshold=threshold,**options)
        print(json.dumps(dict(link_seed=link_seed,threshold=threshold,
            rows=[x for x in result['rows'] if x['method'] in
                ('prefix_gated_mixture','delivered_uncertainty_mixture')],
            costs=result['gate_metrics'],control_delivered=result['control_delivered'],
            mask=result['delivery_mask'],factorial=result['gate_factorial'])),flush=True)
