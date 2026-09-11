"""Moving-target prediction-alignment gate for short-history active sensing."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.audit_short_history_horizon import audit as horizon_audit


def audit(horizon=5):
    truth=np.array([12.,6.,0.])
    cases={
        'matched_cv':truth,
        'stale_stationary':np.zeros(3),
        'opposite_bias':-truth,
    }
    rows={}
    for name,prediction in cases.items():
        result=horizon_audit(horizon,truth,prediction)
        final=result['rows'][-1]
        rows[name]=dict(history_pd=final['history_pd'],instantaneous_pd=final['instantaneous_pd'],
            controlled_pd_delta=final['pd_delta'],history_information=final['history_information'],
            history_actions=final['history_actions'],passes_pd_gate=final['history_passes_pd_gate'])
    matched=rows['matched_cv']; stale=rows['stale_stationary']
    return dict(evidence_class='MOVING_TARGET_HISTORY_ALIGNMENT',horizon=horizon,
        elapsed_s=.8*horizon,target_velocity_mps=truth.tolist(),cases=rows,
        matched_minus_stale_history_pd=matched['history_pd']-stale['history_pd'],
        matched_control_gain=matched['controlled_pd_delta'],
        interpretation=('controlled_pd_delta compares policies under the same moving-target diversity; '
            'matched_minus_stale measures prediction alignment and must not be added to it'),
        limitations=['deterministic known CV truth; no process or measurement noise in belief',
            'analytic coherent Gaussian detection with known covariance',
            'local history only; no communication path or distributed consistency claim'])


if __name__=='__main__': print(json.dumps(audit(),indent=2,allow_nan=False))
