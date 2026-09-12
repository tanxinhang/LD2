"""H0-only replication with fixed policy, no detection-based threshold tuning."""
import json
from tools.audit_unified_receiver_transport import audit


if __name__=='__main__':
    for seed in (770001,780001,790001):
        result=audit(samples=100000,trial_seed=seed,report_gate=True,
            mixture_rollout=True,rollouts=64,rollout_threshold=5.638429914577316,
            link_transition=[[.5770750988142292,.42292490118577075],
                             [.1490857946554149,.8509142053445851]],null_only=True)
        print(json.dumps(result),flush=True)
