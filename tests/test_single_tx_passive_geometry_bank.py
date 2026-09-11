from tools.audit_single_tx_passive_geometry_bank import audit


def test_geometry_bank_is_reproducible_and_physically_bounded():
    first = audit(geometries=20, uavs=4, region_size_m=1130.0, seed=712)
    second = audit(geometries=20, uavs=4, region_size_m=1130.0, seed=712)
    assert first == second
    assert 0.0 <= first["best_single_floor_rate"] <= 1.0
    assert first["best_single_floor_rate"] <= first["all_passive_floor_rate"]
    assert first["best_single_pd_mean"] <= first["all_passive_pd_mean"]
    assert first["paired_pd_gain_ci95"][0] >= 0.0
    assert first["geometries_with_any_dd_alias"] == 0


def test_geometry_bank_records_explicit_nondefault_region_scale():
    report = audit(geometries=20, uavs=4, region_size_m=1130.0, seed=713)
    assert report["region_size_m"] == [1130.0, 1130.0]
    assert report["evidence_class"] == (
        "IDEAL_SINGLE_TX_PASSIVE_RX_DIAGNOSTIC")
