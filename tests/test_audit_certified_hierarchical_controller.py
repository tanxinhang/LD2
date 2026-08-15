from tools.audit_certified_hierarchical_controller import audit


def test_audit_entrypoint_is_importable():
    assert callable(audit)
