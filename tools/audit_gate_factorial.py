"""Reproduce prior failed candidate; attribution only, no fitting to H1."""
import json
from tools.audit_unified_receiver_transport import audit
from uav_isac.physical.report_budget import report_budget_check


def run():
    result=audit(samples=100000,trial_seed=760001,report_gate=True,
        mixture_rollout=True,rollouts=64,rollout_threshold=5.638429914577316,
        frozen_gate_threshold=5.638429914577316,
        link_transition=[[.5770750988142292,.42292490118577075],
                         [.1490857946554149,.8509142053445851]])
    report={k:result[k] for k in ('gate_factorial','gate_attribution','gate_metrics')}
    # Zero is the conservative bound when no independent admission evidence
    # was supplied. Do not fit a resource gate using evaluation stop outcomes.
    report['budget_admission']=report_budget_check(1664,128,1152,0.,1664)
    print(json.dumps(report,indent=2))


if __name__=='__main__': run()
