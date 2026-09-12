from tools.audit_unified_receiver_transport import audit


def test_confirmation_is_conservative_and_attribution_is_partitioned():
    base=audit(samples=2000,window=6,report_gate=True)
    confirmed=audit(samples=2000,window=6,report_gate=True,gate_confirmation=True)
    for seed,metrics in base['gate_metrics'].items():
        other=confirmed['gate_metrics'][seed]
        assert other['stop_fraction']<=metrics['stop_fraction']
        assert other['mean_bits']>=metrics['mean_bits']
    for result in (base,confirmed):
        a=result['gate_attribution']
        assert a['lost_count']==result['gate_comparison']['paired_exact_conservative']['reference_only']
        assert 0<=a['actual_lost_also_lost_at_full_threshold']<=a['lost_count']
        assert 0<=a['lost_without_stop']<=a['lost_count']
        assert result['ack_count']==result['retransmission_count']==0
