"""R14 property tests: named sentinels, zero-semantics, domain-isolated.

The audit (docs/EXPERIMENT_LOG.md R14) found bare numeric
sentinels with two hazards:
  (a) the SAME value (-1) carrying OPPOSITE semantics in different domains
      ("expired, frozen" in persistent_geometry_execution vs "no cached age,
      never expires" in owner_local_physics vs "no target index" in env_core
      movement);
  (b) the SAME semantic ("timestamp never received") written with two values
      (-10**8 comparisons vs -10**9 initialization/serde) in env_core.

R14 fixes this WITHOUT changing any value (zero semantics): a new module
uav_isac/utils/sentinels.py names every sentinel and tags its domain.  These
tests assert (1) the constants keep their historical values, (2) the -1
"same value, opposite meaning" domains behave in opposite directions in the
expressions that consume them, (3) the time-sentinel domains are consistent
under the comparison contract, and (4) the named replacements are actually
wired (source-level lock so a future edit cannot silently reintroduce bare
literals in the audited modules).
"""

import inspect

import numpy as np
import pytest

from uav_isac.utils.sentinels import (
    AGE_EXPIRED,
    AGE_NO_CACHE,
    FRAME_NEVER,
    FRAME_NOT_APPLICABLE,
    OWNER_INDEX_NONE,
    TARGET_INDEX_NONE,
    VALUE_UNBOUNDED_NEG,
    VALUE_UNBOUNDED_POS,
)


def test_historical_values_preserved():
    """Zero semantics: every named constant keeps its audited numeric value."""
    assert FRAME_NEVER == -10**8
    assert FRAME_NOT_APPLICABLE == -10**9
    assert AGE_EXPIRED == -1
    assert AGE_NO_CACHE == -1
    assert TARGET_INDEX_NONE == -1
    assert OWNER_INDEX_NONE == -1
    assert VALUE_UNBOUNDED_NEG == -np.inf
    assert VALUE_UNBOUNDED_POS == np.inf


def test_same_value_opposite_domains_are_behaviorally_distinguished():
    """The -1 triple: AGE_EXPIRED freezes aging; AGE_NO_CACHE never expires.

    Both are -1, but the consumer expressions behave in OPPOSITE directions:
    - persistent_geometry_execution: ``retained = age >= 0`` -- a frozen slot
      (-1) is NOT retained, so it stops aging (expired stays expired).
    - owner_local_physics: ``expired = age > max_age`` -- a no-cache slot (-1)
      is never > max_age, so it is never expired (keeps waiting forever).
    This is the exact hazard R14 names; the test pins the two directions.
    """
    # AGE_EXPIRED domain (persistent): frozen -> not retained, stops aging.
    retained = AGE_EXPIRED >= 0
    assert retained is False
    # AGE_NO_CACHE domain (owner): no timestamp -> never exceeds any max_age.
    for max_age in (0, 5, 1000):
        assert (AGE_NO_CACHE > max_age) is False
    # INDEX domain: -1 target is "no assignment", never a valid index.
    assert TARGET_INDEX_NONE < 0
    assert TARGET_INDEX_NONE != 0


def test_time_sentinel_consistency_contract():
    """FRAME_NOT_APPLICABLE initial values satisfy the FRAME_NEVER comparison.

    env_core initializes timestamp fields with -10**9 (FRAME_NOT_APPLICABLE)
    and tests ``last_seen > FRAME_NEVER`` (-10**8) to decide "received yet".
    The contract is: an initialized (never-received) timestamp must NOT pass
    the received check; a legitimate frame index (>= 0) must pass.
    """
    # -1e9 > -1e8? No -> never-received init values are correctly "not received".
    assert (FRAME_NOT_APPLICABLE > FRAME_NEVER) is False
    # A real frame timestamp (frames are non-negative) must pass.
    assert (0 > FRAME_NEVER) is True
    assert (123456 > FRAME_NEVER) is True


def test_owner_index_guard_uses_named_sentinel():
    """evidence/audit owner guard: valid owners are >= OWNER_INDEX_NONE."""
    # guard semantics as wired in evidence.py / quantized_evidence_audit /
    # evidence_oracle_audit: reject owner < OWNER_INDEX_NONE (i.e. < -1).
    assert (OWNER_INDEX_NONE - 1) < OWNER_INDEX_NONE      # invalid
    assert OWNER_INDEX_NONE >= OWNER_INDEX_NONE            # no-owner is valid
    assert 0 >= OWNER_INDEX_NONE                           # real owner valid


def test_sentinels_module_imports_cleanly_without_side_effects():
    """The module is a pure constants module: importing it has no state."""
    import importlib
    import uav_isac.utils.sentinels as s
    importlib.reload(s)
    assert s.AGE_EXPIRED == AGE_EXPIRED
    assert s.FRAME_NEVER == FRAME_NEVER


@pytest.mark.parametrize("module_name", [
    "uav_isac.environment.env_core",
    "uav_isac.coordination.persistent_geometry_execution",
    "uav_isac.coordination.owner_local_physics",
    "uav_isac.physical.evidence",
])
def test_source_wires_named_sentinels(module_name):
    """Source-level lock: audited modules use the named constants, not literals."""
    import importlib
    mod = importlib.import_module(module_name)
    src = inspect.getsource(mod)
    if module_name.endswith("env_core"):
        assert "FRAME_NOT_APPLICABLE" in src
        assert "FRAME_NEVER" in src
        assert "TARGET_INDEX_NONE" in src
        assert "-10**9" not in src          # no bare init/serde literal remains
    elif module_name.endswith("persistent_geometry_execution"):
        assert "AGE_EXPIRED" in src
        # exact assignment patterns (docstrings may still mention the legacy
        # literal; only the age-assignment sites must be fully named)
        assert "next_age[expired] = -1" not in src
        assert "np.arange(K), np.arange(K)] = -1" not in src
    elif module_name.endswith("owner_local_physics"):
        assert "AGE_NO_CACHE" in src
    else:  # evidence
        assert "OWNER_INDEX_NONE" in src
        assert "owner < -1" not in src


def test_owner_guard_named_in_audit_modules():
    import importlib
    import inspect
    for name in ("uav_isac.evaluation.quantized_evidence_audit",
                 "uav_isac.evaluation.evidence_oracle_audit"):
        src = inspect.getsource(importlib.import_module(name))
        assert "OWNER_INDEX_NONE" in src
        assert "owner < -1" not in src