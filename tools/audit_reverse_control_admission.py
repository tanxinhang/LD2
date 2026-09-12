"""Offline simulated reverse reliability; not runtime feedback or ACK."""
import json
from tools.audit_unified_receiver_transport import audit
from uav_isac.physical.control_admission import delivery_lower_bound,admit_control


if __name__=='__main__':
    for start in (920000,930000):
        outcomes=[audit(samples=1,report_gate=True,stop_only_transport=True,
            link_seed=s)['control_delivered'] for s in range(start,start+400)]
        count=sum(outcomes); lower=delivery_lower_bound(count,len(outcomes))
        print(json.dumps(dict(seed_start=start,successes=count,trials=len(outcomes),
            lower95=lower,admitted=admit_control(lower,128,1152),
            scope='offline simulated traces; fixed geometry; calibration then independent check')),flush=True)
