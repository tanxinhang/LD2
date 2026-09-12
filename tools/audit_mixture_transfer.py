"""Prespecified mismatch/link transfer grid; no parameter tuning per case."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.audit_unified_receiver_transport import audit as case_audit


def audit(samples=30000):
    rows=[]
    for velocity in (0.,3.,6.,12.,18.):
        for link in range(3):
            case_index=len(rows)
            r=case_audit(samples=samples,predicted_velocity_x=velocity,
                link_seed=660000+link,trial_seed=670000+3*case_index)
            methods={x['method']:x for x in r['rows']}
            local=methods['local_uncertainty_mixture']; remote=methods['delivered_uncertainty_mixture']
            rows.append(dict(velocity_error=velocity,link_seed=660000+link,
                local_pd=local['pd'],cooperative_pd=remote['pd'],pfa=remote['pfa'],
                delta=remote['pd']-local['pd'],delivered=sum(r['delivery_mask']),
                bits=r['attempted_bits'],communication_energy_j=r['communication_energy_j']))
    deltas=np.array([r['delta'] for r in rows])
    return dict(rows=rows,mean_delta=float(deltas.mean()),
        negative_cases=int(np.sum(deltas<0)),worst_delta=float(deltas.min()),
        worst_cooperative_pd=min(r['cooperative_pd'] for r in rows),
        scope='15 prespecified grid cases, descriptive transfer audit; not iid population confidence',
        uncertainty_std_mps=6.,quadrature_points=9,samples_per_split=samples)


if __name__=='__main__': print(json.dumps(audit(),indent=2))
