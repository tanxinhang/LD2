import pytest
from tools.audit_otfs_multiframe_fixed_geometry import audit


def test_waveform_energy_and_window_resource_contracts():
    result=audit(20000)
    assert result['template_energy']==pytest.approx(result['waveform_samples'])
    fixed=[r for r in result['rows'] if r['budget']=='fixed_window_energy']
    assert all(r['sensing_energy_j']==pytest.approx(fixed[0]['sensing_energy_j']) for r in fixed)
    for row in result['rows']:
        assert max(row['per_node_rf_w'])<=1.
        assert abs(row['pd']-row['pd_theory'])<.02
        assert row['communication_bits']==0
    assert not result['physical_detection_certified']
