# Independent acquisition phase diagnostic

Same eight schedules, five acquisitions per receiver schedule, fixed 0.15 W,
100000 samples per independent calibration/H0/H1 split. Noise covariance is
the declared receiver-local AR(1) projected-template kernel (rho=0.9).
Signal phases are drawn before whitening. The final decision is terminal only.

| Phase model | Scalar energy P_D | Vector energy P_D | Paired seed delta CI95 |
| --- | ---: | ---: | --- |
| Common | 0.210334 | 0.295558 | [0.010069, 0.174358] |
| Independent per acquisition | 0.027414 | 0.575051 | [0.352001, 0.732193] |

Mean held-out P_FA: scalar 0.00093125, vector 0.00099250. These means do not
certify worst-scenario or population P_FA <= 0.001. No deployment gate passes.

The vector statistic y^H C^-1 y is the GLRT for unrestricted complex mean
in the retained projected observation space. It is not the exact likelihood
for fixed amplitudes with independent uniform phases. Independent phases in
the original observations do not remain independent after whitening.

The contrast confirms destructive loss from scalar coherent pooling. Its
magnitude depends on the assumed high noise memory alongside independent
signal phases: whitening can exploit their different temporal structure.
Independent-phase performance exceeding common-phase performance is not a
universal physical property. DD mismatch still removes energy before fusion.

Cross-node observations are centrally available by assumption, without U2U
costs. Next compare explicit signal phase covariance/amplitude priors under
the same noise covariance, then integrate likelihoods using actual belief
probabilities. A larger hypothesis bank alone cannot close these assumptions.

Reproduce: `python tools/audit_independent_phase_receiver.py`.
