import hashlib

import numpy as np
import pytest

from uav_isac.coordination.local_exchange_oracle import (
    enumerate_local_moves,
    role_first_initial_structure,
)


_GOLDEN = {
    0: (10, "aa4fca78db94e45caddd9f6d6c63c513f4584c35b68aa61d8e63d4075b485e3b"),
    1: (10, "aef141ad9147d1d2f3903ad8070cd56f649c70204abc4faf070ea8f6cf1182cf"),
    2: (10, "b799a3f5d462d233cec139b31dc5112c7fb3edfaeaaff1d40e7f08a723eb8718"),
    3: (10, "ab304177c7feab98c467f9cb120ec3dba4bce0e81122c1005960f225795ad893"),
    4: (12, "904caaff3c4b6eb62f5c989d8d985f4a372a3158379b9c1d4e5d94d0bbfaa73b"),
    5: (10, "950158ff5ed110356c3f45d96e7b2b810e6ebaca949606818f10db17ccf4cd91"),
    6: (12, "93457c3864178c8a3c0e86fdabdac0f385210292e64dc05fdbf9d459a50ca3a7"),
    7: (12, "f21db9d6e4c39e089e093f866bdc2db200f9281fdc789514961ddf10f5e7f0b2"),
    8: (10, "8e0e9ea7d2a11a80b3accb3ac328a5d75c16761f786efaf9310d078d9719faa4"),
    9: (10, "3fa0e6f0a7ffb2b0bf485f420a561f171be5b84172d876e51f5aacd2d9fcf297"),
}


@pytest.mark.parametrize("seed", sorted(_GOLDEN))
def test_target_block_role_prefilter_preserves_exact_move_order(seed):
    """Golden sequence predates the exact outside-role prefilter."""
    rng = np.random.default_rng(seed)
    value = rng.uniform(0.01, 2.0, size=(4, 4, 3))
    mask = rng.random((4, 4, 3)) > 0.2
    mask[np.arange(4), np.arange(4), :] = False
    value = np.where(mask, value, 0.0)
    selected, role, owner = role_first_initial_structure(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=6,
    )
    moves = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5", "N6"),
        target_pair_limit=2,
        reports_per_receiver=6,
        n5_target_mode="proxy_weak",
        n5_weak_target_count=3,
        n5_rebuild_scope="target_block",
    )
    digest = hashlib.sha256()
    for move in moves:
        digest.update(move.kind.encode())
        digest.update(move.selected.tobytes())
        digest.update(move.role.tobytes())
        digest.update(move.owner.tobytes())
    expected_count, expected_digest = _GOLDEN[seed]
    assert len(moves) == expected_count
    assert digest.hexdigest() == expected_digest
