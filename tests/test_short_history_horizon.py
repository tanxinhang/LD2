from tools.audit_short_history_horizon import audit


def test_complete_short_history_is_monotone_and_reports_minimum_gate_horizon():
    result=audit(max_horizon=5); rows=result['rows']
    assert all(rows[i+1]['history_information']>=rows[i]['history_information'] for i in range(4))
    passing=[row['horizon'] for row in rows if row['history_pd']>.8]
    assert result['minimum_history_horizon_for_pd_gate']==(min(passing) if passing else None)
    assert all(row['history_pd']>=row['instantaneous_pd']-1e-12 for row in rows)
    assert result['minimum_history_horizon_for_pd_gate']==5
    assert rows[-1]['elapsed_s']==4.
    assert rows[-1]['history_flight_energy_j']>1000*rows[-1]['history_sensing_rf_energy_j']
