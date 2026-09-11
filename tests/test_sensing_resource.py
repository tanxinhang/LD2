import numpy as np
import pytest

from uav_isac.physical.sensing_resource import (
    ResourceOccupancy,
    SensingMode,
    SensingResourceCaps,
    audit_sensing_resource_conservation,
    certify_target_separable_lp_specialization,
    equivalent_frame_average_power,
    separation_diagnostics,
    separation_is_certified,
)


def _caps() -> SensingResourceCaps:
    return SensingResourceCaps(
        epoch_duration_s=1.0,
        bandwidth_hz=100.0,
        peak_rf_power_w=np.array([1.0]),
        sensing_energy_cap_j=np.array([0.7]),
        total_rf_energy_cap_j=np.array([0.9]),
        maximum_tf_fraction=np.array([1.0]),
    )


def test_orthogonal_signatures_have_identity_separation_gram():
    diagnostic = separation_diagnostics(np.eye(3, dtype=np.complex128))
    assert np.allclose(diagnostic.normalized_whitened_gram, np.eye(3))
    assert diagnostic.mutual_coherence == pytest.approx(0.0)
    assert diagnostic.normalized_identity_error == pytest.approx(0.0)
    assert diagnostic.minimum_eigenvalue == pytest.approx(1.0)
    assert separation_is_certified(
        diagnostic, maximum_mutual_coherence=0.05, minimum_eigenvalue=0.9)


def test_separation_metric_is_scale_invariant_and_detects_redundancy():
    signatures = np.array([[1.0, 1.0], [0.0, 0.1]], dtype=np.complex128)
    scaled = signatures @ np.diag([7.0, 0.2])
    first = separation_diagnostics(signatures)
    second = separation_diagnostics(scaled)
    assert np.allclose(
        first.normalized_whitened_gram,
        second.normalized_whitened_gram,
    )
    assert first.mutual_coherence > 0.99
    assert not separation_is_certified(
        first, maximum_mutual_coherence=0.2, minimum_eigenvalue=0.5)


def test_resource_audit_uses_union_occupancy_and_conserves_energy():
    mask = np.zeros((1, 2, 4, 4), dtype=bool)
    mask[0, 0, :, :2] = True
    mask[0, 1, :, 2:] = True
    occupancy = ResourceOccupancy(
        mode=SensingMode.TARGET_SEPARABLE,
        mask=mask,
        stream_ids=("stream-a", "stream-b"),
        hypothesis_ids=(0, 1),
    )
    audit = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j=np.array([[0.2, 0.3]]),
        stream_power_envelope_w=np.array([[0.4, 0.5]]),
        caps=_caps(),
        communication_energy_j=np.array([0.1]),
        communication_power_envelope_w=np.full((1, 4), 0.1),
    )
    assert audit.passed
    assert audit.sensing_energy_j[0] == pytest.approx(0.5)
    assert audit.total_rf_energy_j[0] == pytest.approx(0.6)
    assert audit.occupied_tf_fraction[0] == pytest.approx(1.0)
    assert audit.occupied_tf_area_hz_s[0] == pytest.approx(100.0)
    assert audit.maximum_concurrent_power_envelope_w[0] == pytest.approx(1.0)


def test_shared_waveform_total_energy_is_not_double_counted():
    mask = np.ones((1, 1, 2, 2), dtype=bool)
    occupancy = ResourceOccupancy(
        mode=SensingMode.COMMON_PROBE,
        mask=mask,
        stream_ids=("common",),
        hypothesis_ids=(None,),
    )
    audit = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j=np.array([[0.4]]),
        stream_power_envelope_w=np.array([[0.5]]),
        caps=_caps(),
        measured_total_rf_energy_j=np.array([0.55]),
        measured_total_rf_power_envelope_w=np.full((1, 2), 0.55),
    )
    assert audit.passed
    assert audit.total_rf_energy_j[0] == pytest.approx(0.55)
    assert audit.energy_accounting_mode == "measured_shared_waveform_total"


def test_mode_label_cannot_claim_target_identity_for_common_probe():
    occupancy = ResourceOccupancy(
        mode=SensingMode.COMMON_PROBE,
        mask=np.ones((1, 1, 1, 1), dtype=bool),
        stream_ids=("common",),
        hypothesis_ids=(0,),
    )
    with pytest.raises(ValueError, match="cannot carry"):
        occupancy.validated()


def test_target_separable_requires_integer_observable_hypothesis_ids():
    occupancy = ResourceOccupancy(
        mode=SensingMode.TARGET_SEPARABLE,
        mask=np.ones((1, 1, 1, 1), dtype=bool),
        stream_ids=("candidate",),
        hypothesis_ids=(1.5,),
    )
    with pytest.raises(ValueError, match="observable hypothesis"):
        occupancy.validated()


def test_zero_energy_caps_are_valid_and_fail_positive_allocation():
    occupancy = ResourceOccupancy(
        mode=SensingMode.COMMON_PROBE,
        mask=np.ones((1, 1, 1, 1), dtype=bool),
        stream_ids=("common",),
        hypothesis_ids=(None,),
    )
    caps = SensingResourceCaps(
        epoch_duration_s=1.0,
        bandwidth_hz=100.0,
        peak_rf_power_w=np.array([1.0]),
        sensing_energy_cap_j=np.array([0.0]),
        total_rf_energy_cap_j=np.array([0.0]),
        maximum_tf_fraction=np.array([1.0]),
    )
    audit = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j=np.array([[0.1]]),
        stream_power_envelope_w=np.array([[0.1]]),
        caps=caps,
    )
    assert not audit.passed
    assert "sensing_energy_cap_exceeded" in audit.failure_reasons
    assert "total_rf_energy_cap_exceeded" in audit.failure_reasons


def test_disabled_zero_resource_caps_are_valid():
    occupancy = ResourceOccupancy(
        mode=SensingMode.COMMON_PROBE,
        mask=np.zeros((1, 1, 1, 1), dtype=bool),
        stream_ids=("disabled",),
        hypothesis_ids=(None,),
    )
    caps = SensingResourceCaps(
        epoch_duration_s=1.0,
        bandwidth_hz=100.0,
        peak_rf_power_w=np.array([0.0]),
        sensing_energy_cap_j=np.array([0.0]),
        total_rf_energy_cap_j=np.array([0.0]),
        maximum_tf_fraction=np.array([0.0]),
    )
    audit = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j=np.array([[0.0]]),
        stream_power_envelope_w=np.array([[0.0]]),
        caps=caps,
    )
    assert audit.passed


def test_nonzero_communication_energy_requires_peak_power_trace():
    occupancy = ResourceOccupancy(
        mode=SensingMode.COMMON_PROBE,
        mask=np.ones((1, 1, 1, 1), dtype=bool),
        stream_ids=("common",),
        hypothesis_ids=(None,),
    )
    with pytest.raises(ValueError, match="requires a power envelope"):
        audit_sensing_resource_conservation(
            occupancy,
            stream_energy_j=np.array([[0.1]]),
            stream_power_envelope_w=np.array([[0.1]]),
            caps=_caps(),
            communication_energy_j=np.array([0.1]),
        )


def test_resource_violations_fail_without_changing_allocation():
    mask = np.ones((1, 2, 1, 1), dtype=bool)
    occupancy = ResourceOccupancy(
        mode=SensingMode.TARGET_SEPARABLE,
        mask=mask,
        stream_ids=("a", "b"),
        hypothesis_ids=(0, 1),
    )
    audit = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j=np.array([[0.6, 0.6]]),
        stream_power_envelope_w=np.array([[0.7, 0.7]]),
        caps=_caps(),
    )
    assert not audit.passed
    assert "peak_rf_power_exceeded" in audit.failure_reasons
    assert "sensing_energy_cap_exceeded" in audit.failure_reasons
    assert "total_rf_energy_cap_exceeded" in audit.failure_reasons


def test_equivalent_power_is_only_energy_over_declared_epoch():
    energy = np.array([[0.01, 0.02]])
    assert np.allclose(equivalent_frame_average_power(energy, 0.1), [[0.1, 0.2]])


def _lp_certificate(
    *,
    mode=SensingMode.TARGET_SEPARABLE,
    signatures=None,
    communication_energy_j=None,
    communication_power_envelope_w=None,
):
    hypothesis_ids = (0, 1) if mode is SensingMode.TARGET_SEPARABLE else (None, None)
    occupancy = ResourceOccupancy(
        mode=mode,
        mask=np.ones((1, 2, 2, 2), dtype=bool),
        stream_ids=("h0", "h1"),
        hypothesis_ids=hypothesis_ids,
    )
    caps = SensingResourceCaps(
        epoch_duration_s=1.0,
        bandwidth_hz=100.0,
        peak_rf_power_w=np.array([0.5]),
        sensing_energy_cap_j=np.array([0.4]),
        total_rf_energy_cap_j=np.array([0.6]),
        maximum_tf_fraction=np.array([1.0]),
    )
    signature_bank = np.eye(2) if signatures is None else signatures
    return certify_target_separable_lp_specialization(
        occupancy,
        stream_energy_j=np.array([[0.2, 0.2]]),
        stream_power_envelope_w=np.array([[0.2, 0.2]]),
        caps=caps,
        hypothesis_signatures=signature_bank,
        maximum_mutual_coherence=0.2,
        minimum_eigenvalue=0.5,
        communication_energy_j=communication_energy_j,
        communication_power_envelope_w=communication_power_envelope_w,
    )


def test_lp_specialization_requires_all_certificates():
    certificate = _lp_certificate()
    assert certificate.passed
    assert np.allclose(certificate.equivalent_average_power_w, [[0.2, 0.2]])


def test_lp_specialization_rejects_mode_label_and_signature_counterexamples():
    common = _lp_certificate(mode=SensingMode.COMMON_PROBE)
    assert not common.passed
    assert "common_probe_has_no_target_separable_lp_mapping" in common.failure_reasons

    redundant = _lp_certificate(signatures=np.array([[1.0, 1.0], [0.0, 0.01]]))
    assert not redundant.passed
    assert "separation_not_certified" in redundant.failure_reasons
    assert redundant.equivalent_average_power_w is None


def test_lp_specialization_counts_communication_peak_power():
    overloaded = _lp_certificate(
        communication_energy_j=np.array([0.2]),
        communication_power_envelope_w=np.full((1, 2), 0.2),
    )
    assert not overloaded.passed
    assert "peak_rf_power_exceeded" in overloaded.failure_reasons
