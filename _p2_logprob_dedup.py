"""P2-1: replace the two byte-duplicated inline log-prob blocks with the
shared helper.

evaluate_actions keeps its computational-graph path; the component head
tensor (movement/message/rate/resource) now comes from the helper's
`return_components` arm.  verify_old_log_prob_consistency calls the same
helper under `no_grad` and compares against stored old log-probs.
"""

import io

PATH = "uav_isac/agents/mappo_agent.py"

EVAL_CALL = """\
        if return_log_prob_components:
            new_log_probs, entropies, lp_components = (
                self._movement_message_resource_log_probs(
                    dp_mean, dp_log_std, role_logits, comm_msgs,
                    actions_dp, actions_role, movement_action_mask,
                    actions_comm, actions_comm_rate,
                    actions_isac_power_raw, actions_sensing_raw,
                    return_components=True))
            head_outputs = {
                'log_probs': torch.stack([
                    lp_components['movement_lp'],
                    lp_components['message_lp'],
                    lp_components['rate_lp'],
                    lp_components['resource_lp'],
                ], dim=-1),
                'credit_values': credit_values,
            }
            return (new_log_probs, values, entropies, dp_mean, pd_pred,
                    comm_msgs, head_outputs)
        new_log_probs, entropies = self._movement_message_resource_log_probs(
            dp_mean, dp_log_std, role_logits, comm_msgs,
            actions_dp, actions_role, movement_action_mask,
            actions_comm, actions_comm_rate,
            actions_isac_power_raw, actions_sensing_raw)
        return new_log_probs, values, entropies, dp_mean, pd_pred, comm_msgs
"""

VERIFY_CALL = """\
            with torch.no_grad():
                dp_mean, dp_log_std, role_logits, comm_msgs, _, _ = self.actor(
                    obs, h_prev, window_mask=window_mask,
                    comm_round_phase=comm_round_phase,
                    agent_identity=agent_identity)
                new_log_probs, _ = self._movement_message_resource_log_probs(
                    dp_mean, dp_log_std, role_logits, comm_msgs,
                    actions_dp, actions_role, movement_action_mask,
                    actions_comm, actions_comm_rate,
                    actions_isac_power_raw, actions_sensing_raw)
            diff = (old_log_probs - new_log_probs).abs()
            max_diff = diff.max().item()
            passed = max_diff < tolerance
            return passed, max_diff
"""


def main() -> None:
    with open(PATH, encoding="utf-8-sig") as fh:
        src = fh.read()

    # --- evaluate_actions inline block ---
    a_start = src.index("\n        # Compute log probs and entropy\n")
    a_end = src.index(
        "\n        return new_log_probs, values, entropies, dp_mean, "
        "pd_pred, comm_msgs", a_start)
    a_end += len("\n        return new_log_probs, values, entropies, "
                 "dp_mean, pd_pred, comm_msgs")
    # sanity: the block must contain the old movement/message/rate literals
    block = src[a_start:a_end]
    assert "log_prob_dp = -0.5 * (" in block or "log_prob_dp = (" in block
    assert "message_lp" in block and "rate_lp" in block
    src = src[:a_start] + EVAL_CALL + src[a_end:]

    # --- verify_old_log_prob_consistency inline block ---
    v_start = src.index("\n            N = obs.shape[0]\n")
    v_end = src.index(
        "\n        passed = max_diff < tolerance\n        return passed, "
        "max_diff", v_start)
    v_end += len("\n        passed = max_diff < tolerance\n        return "
                 "passed, max_diff")
    v_block = src[v_start:v_end]
    assert "log_prob_dp" in v_block and "message_lp" in v_block
    src = src[:v_start] + VERIFY_CALL + src[v_end:]

    with open(PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(src)
    print("P2-1 dedup applied to", PATH)


if __name__ == "__main__":
    main()