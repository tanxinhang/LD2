"""AI may propose candidates but cannot bypass analytical certification."""

import numpy as np
import torch
from torch import nn

from uav_isac.coordination.ai_candidate_screener import certified_ai_screen


class FixedScorer(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


def test_correct_ai_proposal_is_certified_with_a_small_shortlist():
    exact = np.asarray([0.2, 0.9, 0.4, 0.1])
    features = exact[:, None].astype(np.float32)
    upper = exact.copy()
    result = certified_ai_screen(
        FixedScorer(), features, upper,
        lambda indices: exact[indices],
        topk=1, incumbent_index=0,
    )

    assert result.certified
    assert result.chosen_index == 1
    assert len(result.evaluated_indices) == 2


def test_wrong_ai_ranking_is_repaired_by_the_upper_bound_certificate():
    exact = np.asarray([0.2, 0.9, 0.4, 0.1])
    wrong_scores = np.asarray([1.0, 0.0, 0.8, 0.7])[:, None]
    result = certified_ai_screen(
        FixedScorer(), wrong_scores, exact.copy(),
        lambda indices: exact[indices],
        topk=1, incumbent_index=0,
    )

    assert result.certified
    assert result.chosen_index == 1


def test_uncertified_deadline_returns_the_feasible_incumbent():
    exact = np.asarray([0.3, 0.9, 0.4, 0.1])
    wrong_scores = np.asarray([1.0, 0.0, 0.8, 0.7])[:, None]
    result = certified_ai_screen(
        FixedScorer(), wrong_scores, exact.copy(),
        lambda indices: exact[indices],
        topk=1, incumbent_index=0,
        max_exact_evaluations=1,
    )

    assert not result.certified
    assert result.used_incumbent_fallback
    assert result.chosen_index == 0


def test_anytime_budget_executes_best_exactly_verified_candidate():
    exact = np.asarray([0.3, 0.9, 0.4, 0.1])
    scores = np.asarray([0.0, 1.0, 0.8, 0.7])[:, None]
    result = certified_ai_screen(
        FixedScorer(), scores, exact + 1.0,
        lambda indices: exact[indices],
        topk=1, incumbent_index=0,
        max_exact_evaluations=2,
        execute_best_verified_on_budget_exhaustion=True,
    )

    assert not result.certified
    assert not result.used_incumbent_fallback
    assert result.chosen_index == 1
    assert result.chosen_index in result.evaluated_indices


def test_ai_can_never_certify_a_candidate_above_a_valid_upper_bound():
    exact = np.asarray([0.4, 0.8, 0.2])
    features = np.asarray([[1.0], [0.0], [0.5]], dtype=np.float32)
    upper = np.asarray([0.4, 0.8, 0.2])
    result = certified_ai_screen(
        FixedScorer(), features, upper,
        lambda indices: exact[indices],
        topk=1, incumbent_index=0,
    )

    assert result.certified
    assert result.exact_gain == exact[result.chosen_index]


def test_random_wrong_rankings_still_return_the_exact_teacher_best():
    rng = np.random.default_rng(20260827)
    for _ in range(50):
        exact = rng.uniform(0.0, 2.0, size=64)
        upper = exact + rng.uniform(0.0, 0.5, size=64)
        random_scores = rng.normal(size=(64, 1)).astype(np.float32)
        incumbent = int(rng.integers(0, exact.size))
        result = certified_ai_screen(
            FixedScorer(), random_scores, upper,
            lambda indices, values=exact: values[indices],
            topk=4, incumbent_index=incumbent,
        )

        assert result.certified
        assert result.chosen_index == int(np.argmax(exact))


def test_certificate_repairs_are_sent_to_the_verifier_in_batches():
    exact = np.linspace(0.0, 1.0, 129)
    upper = exact + 0.2
    calls: list[int] = []

    def verify(indices: np.ndarray) -> np.ndarray:
        calls.append(int(indices.size))
        return exact[indices]

    result = certified_ai_screen(
        FixedScorer(), -exact[:, None].astype(np.float32), upper, verify,
        topk=1, incumbent_index=0, verification_batch_size=16,
    )

    assert result.certified
    assert result.chosen_index == 128
    assert max(calls[1:]) <= 16
    assert any(size > 1 for size in calls[1:])
