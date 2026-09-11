# Prediction-matched receiver diagnostic

The receiver weights now depend on predicted templates and predicted signal
means only. Truth enters the simulator's projected H1 mean. For real coherent
projections of proper complex Gaussian noise, the covariance is Re(C), and
the true mean is a_true Re(u_pred^H u_true). Negative phase alignment remains
negative. The former oracle-information evaluations did not test this loss.

Eight fixed seeds, five epochs, 100000 samples per calibration/H0/H1 split:

| Metric | Value |
| --- | ---: |
| Calibrated threshold | 3.11121045 |
| Held-out P_FA | 0.00089 |
| Mean prediction-matched P_D | 0.15373125 |
| Worst sampled prediction-matched P_D | 0.00027 |
| Mean ideal matched-reference P_D | 0.98969464 |

This is a severe receiver mismatch failure. High ideal-information scores do
not establish realizable detection. The immediate research priority is target
uncertainty-aware delay/Doppler and phase detection with a calibrated global
false-alarm event, followed by message ownership and transport closure.

Scope remains centralized and coherent, with the declared known AR(1) noise
kernel. Scalar Gaussian sampling exactly evaluates the linear statistic in
that model; it is not a waveform Monte Carlo or field validation. H1 noise
draws are shared across seeds for comparison and must not be treated as
independent population samples. The measured P_FA point estimate below 0.001
is not a confidence certificate. No deployed detection gate is passed.

Reproduce with `python tools/audit_prediction_matched_receiver.py`.
