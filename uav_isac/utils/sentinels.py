"""Named sentinel constants for cross-module semantics (R14, roadmap 2026-08-29).

The audit found bare numeric sentinels scattered across modules with two
hazards: (a) the SAME value used for DIFFERENT semantics in different domains
(the classic ``-1`` triple: "expired-frozen" in persistent_geometry_execution,
"no cached age" in owner_local_physics, "no target index" in env_core
movement), and (b) the SAME semantic expressed with DIFFERENT values (time
"never received" written both as ``-10**8`` and ``-10**9`` in env_core).

R14 fixes this WITHOUT changing any numeric value (zero semantics): every
core occurrence is now named, the meanings are documented here, and the
cross-domain ``-1`` usages are tagged so a future edit cannot silently mix
them.  Values are intentionally NOT unified: changing ``-10**9`` to
``-10**8`` would alter get_state/set_state serialized bytes and replay
hashes; unification is a behaviour change and is deliberately deferred
(roadmap R14 follow-up item).

Domain tags:
- TIME  : frame-timestamp domain (env_core last_seen / update-frame fields).
- AGE   : token/coefficient age domain (persistent inbox, owner cache).
- INDEX : index domain (target/owner identifiers; -1 = "none").
- VALUE : objective/value domain (unbounded LP / candidate scores).
"""

from __future__ import annotations

import numpy as np

# ---- TIME domain (frames) -------------------------------------------------
# "No valid timestamp received yet" -- history debt: written both as -10**8
# (comparison side) and -10**9 (initialization + serde side).  Values kept;
# a true unification (-10**9 everywhere, with comparisons adjusted) changes
# serialized bytes and is deferred (roadmap R14 follow-up).
FRAME_NEVER: int = -10**8          # comparisons: ``last_seen > FRAME_NEVER``
FRAME_NOT_APPLICABLE: int = -10**9  # initialization + get/set_state serde

# ---- AGE domain ------------------------------------------------------------
# persistent_geometry_execution: a token slot whose age exceeded TTL is frozen
# at -1 (no longer aged, mask cleared).  Semantics: EXPIRED.
AGE_EXPIRED: int = -1
# owner_local_physics: no cached target-invariant age available (output field
# target_invariant_age_frames; cache entries are validated non-negative, so -1
# only occurs on the no-cache path).  Semantics: NO_CACHE.
AGE_NO_CACHE: int = -1
# NOTE: AGE_EXPIRED == AGE_NO_CACHE == -1 BY DESIGN of the original code.  The
# two domains never meet in one expression; keep them tagged separately so a
# future edit cannot conflate "frozen as expired" with "no cached age".

# ---- INDEX domain ----------------------------------------------------------
# env_core movement/execution: no target assigned (_distributed_movement_target,
# executed_target, probe_target).  Semantics: NO_ASSIGNMENT.
TARGET_INDEX_NONE: int = -1
# evidence owner: no owner assigned (evidence.py / quantized_evidence_audit
# guard ``owner < OWNER_INDEX_NONE`` rejects invalid owners).  Semantics: NONE.
OWNER_INDEX_NONE: int = -1

# ---- VALUE domain ----------------------------------------------------------
# Unbounded objective/candidate values (LP lb/ub, candidate best-so-far,
# feasibility thresholds).  Standard in argmax/dual contexts.
VALUE_UNBOUNDED_NEG: float = -np.inf
VALUE_UNBOUNDED_POS: float = np.inf