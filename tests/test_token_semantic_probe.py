import numpy as np

from tools.probe_token_semantics import (
    ProbeDataset,
    build_feature_views,
    intervene_semantic,
    save_dataset_npz,
)


def _dataset() -> ProbeDataset:
    metadata = np.asarray([
        [0.0, 0.0, 0.0, 0.25],
        [0.5, 1.0, 0.5, 0.25],
        [1.0, 0.0, 1.0, 0.25],
        [0.0, 1.0, 0.0, 0.25],
    ])
    tokens = np.arange(4 * 16, dtype=np.float64).reshape(4, 16)
    values = np.linspace(0.0, 1.0, 4)
    return ProbeDataset(
        metadata=metadata,
        tokens=tokens,
        current_local_pd=values,
        next_local_pd=values,
        next_team_pd=values,
        distance_m=100.0 + values,
        episode_seed=np.asarray([1, 1, 2, 2]),
    )


def test_feature_views_keep_bid_separate_from_semantics():
    dataset = _dataset()
    views = build_feature_views(dataset)
    assert views["metadata"].shape == (4, 4)
    assert views["contract"].shape == (4, 5)
    assert views["semantic_only"].shape == (4, 15)
    assert views["metadata_semantic"].shape == (4, 19)
    assert views["full"].shape == (4, 20)
    np.testing.assert_allclose(views["contract"][:, -1], dataset.tokens[:, 0])
    np.testing.assert_allclose(views["semantic_only"], dataset.tokens[:, 1:])


def test_zero_semantic_preserves_metadata_and_bid():
    dataset = _dataset()
    full = build_feature_views(dataset)["full"]
    zeroed = intervene_semantic(
        full,
        metadata_dim=dataset.metadata.shape[1],
        mode="zero",
        target_ids=np.asarray([0, 1, 0, 1]),
        seed=7,
    )
    np.testing.assert_allclose(zeroed[:, :5], full[:, :5])
    np.testing.assert_allclose(zeroed[:, 5:], 0.0)


def test_permutation_is_target_conditional_and_preserves_contract():
    dataset = _dataset()
    full = build_feature_views(dataset)["full"]
    target_ids = np.asarray([0, 1, 0, 1])
    permuted = intervene_semantic(
        full,
        metadata_dim=dataset.metadata.shape[1],
        mode="permute",
        target_ids=target_ids,
        seed=9,
    )
    np.testing.assert_allclose(permuted[:, :5], full[:, :5])
    for q in (0, 1):
        idx = np.flatnonzero(target_ids == q)
        original_rows = sorted(map(tuple, full[idx, 5:]))
        permuted_rows = sorted(map(tuple, permuted[idx, 5:]))
        assert permuted_rows == original_rows


def test_dataset_npz_round_trip(tmp_path):
    dataset = _dataset()
    path = tmp_path / "probe.npz"
    save_dataset_npz(path, dataset)
    with np.load(path) as loaded:
        np.testing.assert_allclose(loaded["metadata"], dataset.metadata)
        np.testing.assert_allclose(loaded["tokens"], dataset.tokens)
        np.testing.assert_array_equal(
            loaded["episode_seed"], dataset.episode_seed)
