import numpy as np
import torch

from config.params import load_config
from tools.pretrain_qos_commitment import build_agent
from uav_isac.environment.env_wrapper import UAVISACEnv


def test_observational_semantic_decoder_cannot_change_actions():
    cfg = load_config(
        "config/exp_800_q4_u2u_hierarchical_multistatic_"
        "distributed_matching_hybrid50_semantic_decoder_eval.yaml")
    env = UAVISACEnv(config=cfg, seed=17)
    agent = build_agent(cfg, env, "cpu")
    obs, _ = env.reset(seed=17)
    batch = torch.as_tensor(
        np.stack([obs[str(k)] for k in range(cfg.scenario.K)]),
        dtype=torch.float32,
    )
    identities = torch.arange(cfg.scenario.K, dtype=torch.long)
    with torch.inference_mode():
        before = agent.actor(batch, agent_identity=identities)[:5]
        for parameter in agent.actor.comm_semantic_decoder.parameters():
            parameter.fill_(7.0)
        after = agent.actor(batch, agent_identity=identities)[:5]
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert agent.actor.last_comm_semantic_pd is not None
    env.close()
