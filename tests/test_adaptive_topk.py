import torch

from config.params import load_config
from uav_isac.agents.trainer import build_rate_conditioned_topk_mask
from uav_isac.environment.observation import ObservationBuilder


def test_rate_conditioned_topk_mask_uses_rank_and_cardinality():
    scores = torch.tensor([
        [0.1, 0.9, 0.4, 0.2],
        [0.8, 0.1, 0.7, 0.3],
        [0.2, 0.5, 0.9, 0.4],
    ])
    rates = torch.tensor([0, 1, 2])
    mask = build_rate_conditioned_topk_mask(
        scores, rates, [0, 1, 2])

    torch.testing.assert_close(
        mask.sum(dim=-1), torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(mask[0], torch.zeros(4))
    torch.testing.assert_close(mask[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(mask[2], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_rate_conditioned_topk_respects_warm_start_support():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    maximum = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    mask = build_rate_conditioned_topk_mask(
        scores, torch.tensor([2]), [0, 1, 2], maximum_mask=maximum)
    # Ranking happens inside the two-edge warm-start support. The globally
    # highest but unsupported edge must never be exposed.
    torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))


def test_adaptive_topk_config_separates_bits_from_token_count():
    cfg = load_config(
        'config/exp_800_q4_u2u_hierarchical_multistatic_'
        'distributed_matching_hybrid50_adaptive_topk12.yaml')
    assert cfg.marl.adaptive_topk_from_rate_enabled is True
    assert cfg.marl.comm_rate_bits_per_dim == [0, 8, 8]
    assert cfg.marl.adaptive_topk_rate_mapping == [0, 1, 2]
    assert cfg.marl.comm_rate_metadata_denominator == 4.0
    assert cfg.marl.sparse_claim_share_topk == 2
    assert cfg.marl.comm_rate_bonus_aux_coef == 0.0
    assert cfg.marl.adaptive_topk_rate_only_training is True


def test_rate_metadata_can_preserve_legacy_normalization():
    builder = ObservationBuilder(
        K=4, Q=4, comm_num_rate_levels=3,
        comm_rate_metadata_denominator=4.0)
    assert builder.comm_rate_metadata_denominator == 4.0
    # Warm-started active index two retains the historical 2/(5-1)=0.5.
    assert 2.0 / builder.comm_rate_metadata_denominator == 0.5
