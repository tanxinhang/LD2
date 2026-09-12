# Independent H0-only calibration stability

The rejected gate is frozen at forecast threshold 5.638429914577316 with
64 rollouts and the previously trained observed-link transition matrix.
Three independent 100,000-sample H0 splits are used, seeds 770001, 780001,
790001. No H1 samples are generated, no candidate is selected by detection
results, and the policy is not retuned between replications.

Run `python -m tools.audit_null_calibration_stability`.

The audit now has `null_only=True`, avoiding unnecessary H1 simulation and
returning a fixed-policy order-statistic tail bound. With ascending zero-based
index k=ceil((n-1)*0.9995), a continuous iid H0 score distribution yields
tail mass above that order statistic distributed as Beta(n-k,k+1). Its 95th
percentile bounds the unknown false-alarm probability with 95% calibration
coverage for one prespecified policy/replication. At n=100,000 the upper bound
is 0.000621670, versus 0.000915116 at n=10,000.

These are pointwise bounds, not simultaneous bounds over candidate selection,
nor population coverage over channel/geometry. They bound false alarms above,
not how close calibration is to the target; a very conservative threshold can
still satisfy the bound while sacrificing detection. The earlier adaptive
fixed-point search on reused H0 data does not inherit this guarantee.

Replication thresholds:

- Seed 770001: 4.0558007150.
- Seed 780001: 3.9908186417.
- Seed 790001: gated 4.1563693073, full 4.1653969635.

The first two have equal gated/full thresholds. All three are substantially
below the previous 5.6384299146. This is direct H0
evidence of calibration instability, complementing the paired H1 attribution.
It is not evidence that the gate saves resources or that rollout decisions
have converged. No new PD claim is made in this H0-only experiment.

Do not silently deploy these thresholds: the frozen forecasting policy still
uses 5.6384299146. A new policy/threshold pair requires independent design and
calibration, followed by new H0/H1 validation and resource admission. All gate
variants remain default off; full reporting remains the operational reference.
