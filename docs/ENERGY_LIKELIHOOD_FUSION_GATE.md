# Prediction-based energy likelihood fusion

For normalized projection energy e, deterministic signal magnitude sqrt(snr)
and unknown deterministic phase, the phase-invariant log density ratio is
log I0(2 sqrt(snr e))-snr. No phase probability prior is introduced.
The result is exact for the declared energy distribution, not for arbitrary
SNR uncertainty or correlated acquisitions. The receiver uses predicted SNR.

Same 13-frame frozen experiment, 100000 trials per split:

| Rule | P_D | P_FA |
| --- | ---: | ---: |
| Local energy sum | 0.82900 | 0.00046 |
| Delivered energy sum | 0.80748 | 0.00040 |
| Local energy likelihood | 0.85181 | 0.00044 |
| Delivered energy likelihood | 0.81020 | 0.00046 |

Each final statistic is calibrated independently; 14 simultaneous intervals
replace the previous ten after adding two rules. This is a conditional model
comparison on one channel trace. The improvement of local likelihood does
not establish a population or multi-geometry advantage.

Remote predicted SNR ignores actual template mismatch, so delivered likelihood
remains inferior to local-only detection. Energy payloads, quantization, FBL
delivery masks and power allocations are unchanged. Prediction-derived SNR
is preconfigured at the receiver; distributing/updating that model would need
explicit communication accounting. Truth overlap is never used for weighting.

Next investigate receiver-visible uncertainty in effective SNR/template
alignment before source selection. Dropping a source using true overlap would
repeat the oracle leak. This audit does not justify such an implementation.
