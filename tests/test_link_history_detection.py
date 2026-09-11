from tools.audit_link_history_detection import audit


def test_actual_link_delivery_and_silence_drive_history():
    result=audit(trials=10000)
    silent=result['rows'][0]
    assert silent['mean_attempted_bits']==0
    assert silent['delivered_remote_blocks']==[0]*8
    assert silent['pd']==silent['local_history_pd']
    assert max(result['rows'][-1]['delivered_remote_blocks'])<=8
    assert sum(result['rows'][-1]['delivered_remote_blocks'])>0
    assert result['rows'][-1]['mean_report_tx_energy_j']>0
    assert all(max(row['per_node_power_w'])<=1 for row in result['rows'])
