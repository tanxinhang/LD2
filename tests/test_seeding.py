"""Determinism of uav_isac.utils.seeding.set_seed.

utils/seeding.py was the one module with zero direct test references (all other
core modules have dedicated coverage). These tests pin its contract: a seed
must deterministically reset the three global RNG streams and must not error
on a CPU-only machine.
"""

import random
import numpy as np
import pytest
import torch

from uav_isac.utils.seeding import set_seed


@pytest.fixture(autouse=True)
def _restore_global_rng_state():
    """Isolate set_seed's global side effects from the rest of the suite."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


def _draw_triplet():
    py = [random.random() for _ in range(5)]
    npx = np.random.random(5).tolist()
    tx = torch.rand(5).tolist()
    return (py, npx, tx)


def test_set_seed_is_repeatable_for_all_three_streams():
    set_seed(20260904)
    first = _draw_triplet()
    set_seed(20260904)
    second = _draw_triplet()
    assert first == second


def test_set_seed_changes_all_three_streams_with_different_seed():
    set_seed(1)
    a = _draw_triplet()
    set_seed(2)
    b = _draw_triplet()
    assert a != b, "different seeds produced identical global RNG draws"


def test_set_seed_does_not_raise_without_cuda():
    # Should succeed on CPU-only runs even though the CUDA branches are gated.
    set_seed(1234)
    assert torch.rand(3).shape == (3,)


def test_repeated_seed_yields_identical_torch_sequence():
    set_seed(99)
    t1 = torch.rand(8).tolist()
    set_seed(99)
    t2 = torch.rand(8).tolist()
    assert t1 == t2
