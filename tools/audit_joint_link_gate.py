"""Disjoint link fitting, threshold design and frozen detection validation."""
import json
import numpy as np
from tools.audit_unified_receiver_transport import audit
from uav_isac.physical.observable_link_predictor import fit_delivery_markov


def run():
    def traces(first):
        return np.asarray([audit(samples=16,report_gate=True,link_seed=s)['delivery_mask']
            for s in range(first,first+80)],bool)
    training=traces(820000); heldout=traces(830000)
    transition=fit_delivery_markov(training)
    probability=transition[heldout[:,:-1].astype(int),1]
    iid=(training.sum()+1)/(training.size+2)
    diagnostics=dict(transition=transition.tolist(),
        heldout_markov_brier=float(np.mean((probability-heldout[:,1:])**2)),
        heldout_iid_brier=float(np.mean((iid-heldout[:,1:])**2)))
    print(json.dumps(diagnostics),flush=True)
    threshold=4.322597843473792; iterations=[]
    # This is a bounded fixed-point experiment, not guaranteed convergence.
    for step in range(4):
        result=audit(samples=10000,trial_seed=750001,report_gate=True,
            mixture_rollout=True,rollouts=64,rollout_threshold=threshold,
            link_transition=transition)
        calibrated=next(x['threshold'] for x in result['rows'] if x['method']=='prefix_gated_mixture')
        iterations.append(dict(step=step,forecast=threshold,calibrated=calibrated))
        print(json.dumps(iterations[-1]),flush=True)
        if abs(calibrated-threshold)<1e-10: break
        threshold=calibrated
    # Freeze BOTH at one common design threshold. Held-out false alarms decide
    # whether it is acceptable; do not silently recalibrate on validation.
    result=audit(samples=100000,trial_seed=760001,report_gate=True,
        mixture_rollout=True,rollouts=64,rollout_threshold=threshold,
        frozen_gate_threshold=threshold,link_transition=transition)
    print(json.dumps(dict(design=iterations,link=diagnostics,
        rows=[x for x in result['rows'] if x['method'] in ('prefix_gated_mixture','delivered_uncertainty_mixture')],
        comparison=result['gate_comparison'],cost=result['gate_metrics'],
        scope='single held-out sensing seed and frozen channel; design uses H0 only')),flush=True)


if __name__=='__main__': run()
