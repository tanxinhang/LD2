from tools.audit_joint_structure_relaxation import (
    _exact_reference_rows,
)


def test_optional_exact_reference_is_empty():
    assert _exact_reference_rows(None) == {}
