def test_replay_is_explicitly_development_only():
    # The full trace replay is intentionally an integration audit.  This unit
    # test locks the claim boundary directly in the tool source/API contract.
    source = __import__(
        "inspect").getsource(__import__(
            "tools.audit_joint_structural_locator_replay",
            fromlist=["replay"]))
    assert '"evidence_class": "DEVELOPMENT_ONLY_NOT_BLIND"' in source
    assert '"blind_generalization"' in source
