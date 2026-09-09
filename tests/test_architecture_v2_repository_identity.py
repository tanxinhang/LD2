from uav_isac.adapters import build_repository_identity


def test_repository_identity_is_repeatable_and_covers_dirty_source():
    first = build_repository_identity()
    second = build_repository_identity()

    assert first == second
    assert len(first.commit) == 40
    assert len(first.sha256) == 64
    assert first.dirty

