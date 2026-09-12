"""Independent calibration of STOP-only slot-keyed detector branches."""
import json
from tools.audit_unified_receiver_transport import audit


if __name__=='__main__':
    options=dict(report_gate=True,stop_only_transport=True,mixture_rollout=True,
        rollouts=64,rollout_threshold=4.416818344356382,
        link_transition=[[.5770750988142292,.42292490118577075],
                         [.1490857946554149,.8509142053445851]])
    calibration=audit(samples=100000,trial_seed=870001,null_only=True,**options)
    threshold=calibration['thresholds']['prefix_gated_mixture']
    print(json.dumps(dict(stage='calibration',threshold=threshold)),flush=True)
    result=audit(samples=100000,trial_seed=880001,frozen_gate_threshold=threshold,**options)
    print(json.dumps({k:result[k] for k in ('rows','gate_metrics','gate_comparison',
        'delivery_mask','control_delivered','stop_only_transport')},indent=2),flush=True)
