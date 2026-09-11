import numpy as np
from tools.audit_noack_markov_fusion import model,trace,links


def test_once_only_link_billing_and_single_block_recursions_agree():
    positions,grams=model()
    mask,energy,bits=links(positions,120001,blocks=2)
    assert bits==2*258  # exactly two fresh packets, no ACK/retry
    assert len(mask)==2 and energy>=0
    scores=trace(grams,[True,False],256,1,99)
    assert np.allclose(scores[1]['markov'],scores[1]['memoryless'])
    assert np.all(np.isfinite(scores[2]['markov']))


def test_all_missing_remote_emissions_reduce_exactly_to_local_history():
    _,grams=model()
    scores=trace(grams,[False]*8,256,0,99)
    for row in scores.values():
        assert np.allclose(row['markov'],row['local'])
