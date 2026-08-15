import numpy as np
import pytest

from uav_isac.evaluation.horizon_transition_gate import (
    CertificateRiskBudget,
    HorizonTransitionBounds,
    certify_horizon_paired_deflection_transition,
    certify_horizon_transition,
    certify_risk_budgeted_horizon,
    risk_budgeted_certificate_union,
    validate_resource_epoch_risk_budget,
)


def _bounds(source: str, *, miscoverage: float = 0.02):
    return HorizonTransitionBounds(
        source=source,
        candidate_lower=np.asarray([
            [0.62, 0.66],
            [0.64, 0.67],
            [0.66, 0.68],
        ]),
        noop_lower=np.asarray([
            [0.60, 0.65],
            [0.61, 0.65],
            [0.62, 0.65],
        ]),
        noop_upper=np.asarray([
            [0.61, 0.66],
            [0.62, 0.66],
            [0.63, 0.66],
        ]),
        miscoverage=miscoverage,
    )


def test_risk_budget_rejects_unfunded_data_dependent_union():
    with pytest.raises(ValueError, match="exceeds"):
        CertificateRiskBudget(
            total=0.05, feedback=0.025, physics=0.025, link=0.01)

    budget = CertificateRiskBudget(
        total=0.05, feedback=0.02, physics=0.02, link=0.01)
    decision = risk_budgeted_certificate_union(
        {"feedback": False, "physics": True},
        risk_budget=budget,
        hard_feasible=True,
    )
    assert decision.accept
    assert decision.accepted_sources == ("physics",)
    assert np.isclose(decision.allocated_miscoverage, 0.05)


def test_system_risk_budget_charges_link_runtime_and_energy_intersection():
    budget = CertificateRiskBudget(
        total=0.10,
        feedback=0.02,
        physics=0.02,
        link=0.02,
        runtime=0.02,
        energy=0.02,
    )
    assert budget.allocated == pytest.approx(0.10)
    assert budget.allocation_for("runtime") == pytest.approx(0.02)
    assert budget.allocation_for("energy") == pytest.approx(0.02)
    with pytest.raises(ValueError, match=r"runtime \+ energy"):
        CertificateRiskBudget(
            total=0.05,
            feedback=0.02,
            physics=0.02,
            link=0.01,
            runtime=0.01,
            energy=0.01,
        )


def test_resource_epoch_risk_ledger_is_complete_without_independence():
    budget = CertificateRiskBudget(
        total=0.10,
        feedback=0.02,
        physics=0.02,
        link=0.02,
        runtime=0.02,
        energy=0.02,
    )
    ledger = validate_resource_epoch_risk_budget(
        budget,
        link_miscoverage=0.02,
        runtime_miscoverage=0.01,
        energy_miscoverage=0.015,
    )
    assert ledger.complete
    assert ledger.allocated_resource_miscoverage == pytest.approx(0.06)
    assert ledger.joint_coverage_floor_union_bound == pytest.approx(0.90)


def test_resource_epoch_risk_ledger_fails_closed_for_missing_or_oversized_epoch():
    budget = CertificateRiskBudget(
        total=0.10,
        feedback=0.02,
        physics=0.02,
        link=0.02,
        runtime=0.02,
        energy=0.02,
    )
    incomplete = validate_resource_epoch_risk_budget(
        budget,
        link_miscoverage=0.02,
        runtime_miscoverage=None,
        energy_miscoverage=0.02,
    )
    assert not incomplete.complete
    with pytest.raises(ValueError, match="runtime epoch miscoverage exceeds"):
        validate_resource_epoch_risk_budget(
            budget,
            link_miscoverage=0.02,
            runtime_miscoverage=0.021,
            energy_miscoverage=0.02,
        )


def test_horizon_gate_requires_targetwise_no_harm_at_every_step():
    safe = certify_horizon_transition(
        _bounds("physics"), qos_floor=0.60, discount=0.95)
    assert safe.accept
    assert np.all(safe.target_safe)
    assert safe.discounted_gain_lower > 0.0

    unsafe_bounds = _bounds("physics")
    candidate = unsafe_bounds.candidate_lower.copy()
    candidate[1, 1] = 0.40
    unsafe = certify_horizon_transition(
        HorizonTransitionBounds(
            source="physics",
            candidate_lower=candidate,
            noop_lower=unsafe_bounds.noop_lower,
            noop_upper=unsafe_bounds.noop_upper,
            miscoverage=unsafe_bounds.miscoverage,
        ),
        qos_floor=0.60,
    )
    assert not unsafe.accept
    assert not unsafe.target_safe[1, 1]

    lower_only = _bounds("physics")
    candidate = lower_only.candidate_lower.copy()
    candidate[0, 1] = 0.655
    rejected = certify_horizon_transition(
        HorizonTransitionBounds(
            source="physics",
            candidate_lower=candidate,
            noop_lower=lower_only.noop_lower,
            noop_upper=lower_only.noop_upper,
            miscoverage=lower_only.miscoverage,
        ),
        qos_floor=0.60,
    )
    assert candidate[0, 1] >= lower_only.noop_lower[0, 1]
    assert not rejected.target_safe[0, 1]


def test_risk_budgeted_horizon_union_and_hard_transport_gate():
    budget = CertificateRiskBudget(
        total=0.05, feedback=0.02, physics=0.02, link=0.01)
    decision = certify_risk_budgeted_horizon(
        [_bounds("feedback"), _bounds("physics")],
        risk_budget=budget,
        qos_floor=0.60,
        discount=0.95,
        objective_cost=0.001,
        mode="union",
    )
    assert decision.accept
    assert decision.accepted_sources == ("feedback", "physics")

    blocked = certify_risk_budgeted_horizon(
        [_bounds("feedback"), _bounds("physics")],
        risk_budget=budget,
        qos_floor=0.60,
        transport_feasible=False,
        mode="union",
    )
    assert not blocked.accept


def test_route_calibration_must_fit_its_frozen_allocation():
    budget = CertificateRiskBudget(
        total=0.05, feedback=0.02, physics=0.02, link=0.01)
    with pytest.raises(ValueError, match="exceeds its allocation"):
        certify_risk_budgeted_horizon(
            [_bounds("feedback", miscoverage=0.025)],
            risk_budget=budget,
            qos_floor=0.60,
        )


def test_paired_gate_keeps_common_uncertainty_instead_of_crossing_extrema():
    bounds = HorizonTransitionBounds(
        source="physics",
        candidate_lower=np.asarray([[0.70, 0.72]]),
        noop_lower=np.asarray([[0.65, 0.68]]),
        noop_upper=np.asarray([[0.90, 0.91]]),
        miscoverage=0.02,
    )
    legacy = certify_horizon_transition(bounds, qos_floor=0.60)
    paired = certify_horizon_paired_deflection_transition(
        bounds,
        np.asarray([[0.10, 0.05]]),
        qos_floor=0.60,
    )

    assert not legacy.accept
    assert paired.accept
    assert np.all(paired.target_safe)
    assert paired.certificate_mode == "common_box_paired_deflection"
    assert paired.gain_unit == "deflection"


def test_paired_gate_rejects_one_target_deflection_harm():
    paired = certify_horizon_paired_deflection_transition(
        _bounds("physics"),
        np.asarray([
            [0.10, 0.05],
            [0.10, -1.0e-3],
            [0.10, 0.05],
        ]),
        qos_floor=0.60,
    )

    assert not paired.accept
    assert not paired.target_safe[1, 1]
