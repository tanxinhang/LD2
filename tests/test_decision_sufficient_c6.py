"""C6 decision-sufficient comm: split-flag wiring regression tests.

Audit advice/001 §12-13 (2026-08-26).  The C6 decision-sufficient U2U gate is
now a two-stage, fail-closed flag chain instead of one merged switch:

    adaptive_bits  ->  event_trigger  ->  comm_enabled  ->  no_truth_fail_closed

The scalar order-preservation primitives remain available, but the environment
must reject their use on arbitrary learned ``target_tokens``: a distance-score
margin does not bound a neural decoder's action change.  Exact-bit transport is
tested independently and remains available for a future explicit score-token
protocol with a downstream decision certificate.

All flags default OFF.  The canonical manifest also keeps the online chain OFF
until an explicit payload-to-action certificate exists.
"""

import numpy as np
import pytest

from config.params import get_default_config, load_config
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.coordination.structure_regret import (
    decision_sufficient_plan,
    local_decision_margin,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strict_c6_config(adaptive: bool, ladder=(0, 8), max_bits=8):
    """Small strict no-truth decision-sufficient environment config."""
    cfg = get_default_config()
    cfg.scenario.K = 2
    cfg.scenario.Q = 2
    cfg.scenario.T = 3
    cfg.marl.tracking_enabled = True
    cfg.marl.ground_communication_enabled = False
    cfg.marl.distributed_coordination_use_local_belief_targets = True
    cfg.marl.distributed_no_truth_fail_closed = True
    cfg.marl.distributed_decision_sufficient_comm_enabled = True
    cfg.marl.distributed_decision_sufficient_event_trigger_enabled = True
    cfg.marl.distributed_decision_sufficient_adaptive_bits_enabled = adaptive
    cfg.marl.distributed_decision_sufficient_dynamic_range = 1.0
    cfg.marl.distributed_decision_sufficient_max_bits = max_bits
    cfg.marl.comm_rate_bits_per_dim = list(ladder)
    cfg.marl.comm_header_bits = 64
    cfg.marl.comm_bandwidth_hz = 100000
    cfg.marl.comm_deadline_s = 0.005
    cfg.marl.comm_processing_delay_s = 2.0e-4
    cfg.marl.comm_snr_threshold_db = 0.0
    cfg.marl.comm_antenna_gain_dbi = 0.0
    cfg.marl.comm_tx_power_w = 0.25
    cfg.marl.comm_payload_mode = "target_tokens"
    return cfg


def _place_belief_targets(env, sender, distances):
    """Put local-belief targets at ``distances`` [m] along +x from the sender."""
    x = np.asarray(env.uavs[int(sender)].pos[:2], dtype=np.float64)
    for q, d in enumerate(distances):
        env.belief_mgr.mean[int(sender), q, :2] = x + np.array([d, 0.0])


def _submit_pending(env, sender=0):
    env._pending_comm_messages = {
        int(sender): np.ones(env._comm_payload_dim, dtype=np.float64)}
    env._pending_comm_rates = {int(sender): 1}
    env._pending_comm_token_masks = {
        int(sender): np.ones(env.Q, dtype=np.float64)}


def _install_common_reference(env, sender=0, bits=8):
    """Give every peer a live retained token at the declared precision."""
    sender = int(sender)
    for receiver in range(env.K):
        if receiver == sender:
            continue
        env._received_comm_msgs.setdefault(receiver, {})[sender] = np.ones(
            env._comm_payload_dim, dtype=np.float64)
        env._received_comm_meta.setdefault(receiver, {})[sender] = {
            "rate_index": 1,
            "bits_per_dim": int(bits),
            "age_frames": 0,
            "sent_frame": int(env.t),
        }


# ---------------------------------------------------------------------------
# configuration: split flags, pins, fail-closed chain
# ---------------------------------------------------------------------------

def test_c6_split_flags_default_off_keeps_legacy():
    cfg = get_default_config()
    assert cfg.marl.distributed_decision_sufficient_comm_enabled is False
    assert (cfg.marl.distributed_decision_sufficient_event_trigger_enabled
            is False)
    assert (cfg.marl.distributed_decision_sufficient_adaptive_bits_enabled
            is False)


def test_uncertified_c6_split_flags_disabled_in_manifest():
    cfg = load_config("config/system_manifest.yaml")
    assert cfg.marl.distributed_decision_sufficient_comm_enabled is False
    assert (cfg.marl.distributed_decision_sufficient_event_trigger_enabled
            is False)
    assert (cfg.marl.distributed_decision_sufficient_adaptive_bits_enabled
            is False)


def test_c6_fail_closed_chain_constructor():
    base = get_default_config()
    base.scenario.K = 2
    base.scenario.Q = 2
    base.marl.tracking_enabled = True
    base.marl.distributed_coordination_use_local_belief_targets = True

    # comm without the strict no-truth gate -> fail closed.
    cfg_a = get_default_config()
    cfg_a.scenario.K = 2
    cfg_a.scenario.Q = 2
    cfg_a.marl.tracking_enabled = True
    cfg_a.marl.distributed_coordination_use_local_belief_targets = True
    cfg_a.marl.distributed_no_truth_fail_closed = False
    cfg_a.marl.distributed_decision_sufficient_comm_enabled = True
    with pytest.raises(ValueError, match="no_truth_fail_closed"):
        EnvironmentCore(cfg_a)

    # event trigger without comm -> fail closed.
    cfg_b = get_default_config()
    cfg_b.scenario.K = 2
    cfg_b.scenario.Q = 2
    cfg_b.marl.tracking_enabled = True
    cfg_b.marl.distributed_coordination_use_local_belief_targets = True
    cfg_b.marl.distributed_no_truth_fail_closed = True
    cfg_b.marl.distributed_decision_sufficient_comm_enabled = False
    cfg_b.marl.distributed_decision_sufficient_event_trigger_enabled = True
    with pytest.raises(ValueError, match="comm_enabled"):
        EnvironmentCore(cfg_b)

    # adaptive bits without event trigger -> fail closed.
    cfg_c = get_default_config()
    cfg_c.scenario.K = 2
    cfg_c.scenario.Q = 2
    cfg_c.marl.tracking_enabled = True
    cfg_c.marl.distributed_coordination_use_local_belief_targets = True
    cfg_c.marl.distributed_no_truth_fail_closed = True
    cfg_c.marl.distributed_decision_sufficient_comm_enabled = True
    cfg_c.marl.distributed_decision_sufficient_adaptive_bits_enabled = True
    with pytest.raises(ValueError, match="event_trigger"):
        EnvironmentCore(cfg_c)

    # A syntactically complete chain is still mathematically unsupported for
    # learned target tokens: no decoder/action Lipschitz certificate exists.
    full = _strict_c6_config(adaptive=True)
    with pytest.raises(ValueError, match="downstream decision certificate"):
        EnvironmentCore(full)


# ---------------------------------------------------------------------------
# transport: exact_bits_per_dim replaces the rate ladder
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def comm_model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4, 8],
        header_bits=64,
        bandwidth_hz=1.0e5,
        deadline_s=0.005,
        processing_delay_s=2.0e-4,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
        message_dim=16,
    )


def test_transmit_exact_bits_charges_exact_precision(comm_model):
    pos = np.zeros((3, 3))
    out, stats = comm_model.transmit(
        {0: np.ones(16)}, {0: 1}, pos, exact_bits_per_dim={0: 3})
    assert stats.active_senders == 1
    assert stats.total_bits == 64 + 16 * 3  # header + 3 bits/dim
    # ladder path unchanged: rate index 1 -> 4 bits/dim
    out2, stats2 = comm_model.transmit({0: np.ones(16)}, {0: 1}, pos)
    assert stats2.total_bits == 64 + 16 * 4


def test_transmit_exact_bits_quantizes_at_exact_precision(comm_model):
    pos = np.zeros((3, 3))
    msg = 0.3 * np.ones(16)
    out, _ = comm_model.transmit(
        {0: msg}, {0: 1}, pos, exact_bits_per_dim={0: 1})
    # 1 bit/dim -> {-1, +1} levels; 0.3 -> +1
    assert np.all(np.abs(np.asarray(out[0].message)) == 1.0)


def test_transmit_exact_bits_zero_gives_silence(comm_model):
    pos = np.zeros((3, 3))
    out, stats = comm_model.transmit(
        {0: np.ones(16)}, {0: 1}, pos, exact_bits_per_dim={0: 0})
    assert stats.active_senders == 0
    assert out == []


# ---------------------------------------------------------------------------
# env_core wiring: event-triggered silence, adaptive bits on air, legacy OFF
# ---------------------------------------------------------------------------

def test_env_rejects_score_proxy_certificate_for_learned_tokens():
    cfg = _strict_c6_config(adaptive=True)
    assert cfg.marl.comm_payload_mode == "target_tokens"
    with pytest.raises(ValueError, match="downstream decision certificate"):
        EnvironmentCore(cfg)


def test_pilot_disables_uncertified_c6_chain():
    """The strict pilot must not claim score-order guarantees for neural tokens."""
    cfg = load_config("config/exp_strict_distributed_no_truth_pilot.yaml")
    assert cfg.marl.tracking_enabled is True
    assert cfg.marl.ground_communication_enabled is False
    assert cfg.marl.distributed_coordination_use_local_belief_targets is True
    assert cfg.marl.distributed_no_truth_fail_closed is True
    assert cfg.marl.distributed_decision_sufficient_comm_enabled is False
    assert (cfg.marl.distributed_decision_sufficient_event_trigger_enabled
            is False)
    assert (cfg.marl.distributed_decision_sufficient_adaptive_bits_enabled
            is False)
    assert cfg.marl.comm_payload_mode == "target_tokens"
    assert cfg.marl.comm_rate_bits_per_dim == [0, 4]
