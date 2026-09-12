# Noncoherent geometry screening: staged validation

## Model correction

For proper complex noise n~CN(0,C) and z=mu+n:

    mean_H1(|z|^2)-mean_H0(|z|^2) = |mu|^2
    Sigma_E,H0[i,j] = |C[i,j]|^2
    d |mu|^2 = 2 Re(conj(mu) d mu)
    d Sigma_E,H0 = 2 Re(conj(C) d C).

`noncoherent_information.energy_moment_model` implements this conversion and
derivatives. The previous conditional-Deflection gradient can now operate on
actual energy moments. These energies are non-Gaussian; H1 covariance also
depends on signal and phase. No Gaussian ROC formula or phase prior is used.
An independent Monte Carlo validates the frozen linear energy detector.

## Common physical setup

Three nodes near 800m bistatic legs; one transmitter at 0.15 W, two receivers.
Four history acquisitions at initial geometry, then nine at selected geometry.
One planar move <=2.5m, v_max=25m/s and dt=0.1s, fixed altitude20m, endpoint
separation>=20m. Generated nodes are far apart; trajectories and motion energy
are not generally certified by this diagnostic. Total sensing energy is
0.15*13*0.001024=0.0019968J. Roles remain half duplex during each acquisition.

Forecast target is (0,30,0)m with vx=6m/s; held-out target is stationary at
origin. Waveform projection mismatch enters alternative observations. Templates
and weights use forecasts only. History and future receiver IDs are distinct
acquisition metadata; changing role does not transfer a receiver's noise history
to another UAV. Same-receiver temporal noise rho=0 or0.6 is a declared model;
cross-receiver noise is independent. Noise variance per acquisition stays fixed.

Detector is w^T(E-diag(C)), w=Sigma_E,H0^-1 delta_forecast. A phase sequence
1.7*t plus receiver-position-in-vector offset0.3 is declared for H1 generation;
the detector and action score do not consume those phases. It is one unknown
phase scenario, not robustness to every possible phase or tracking model.

Each case uses 50k independent calibration H0, validation H0 and validation H1
samples. Candidate comparisons share standardized Gaussian innovations under
different covariance factors (a counterfactual coupling, not one common raw IQ
record for all actions). Thresholds target PFA=.0005 but achieved PFA differs;
paired deltas are procedure comparisons, not exact equal-PFA ROC superiority.
Confidence intervals are within-case only. Four geometries at two rho values
are not eight independent geometry episodes or a population certification.

Cross-node evidence is ideally pooled. No U2U costs, role-command latency or
flight energy are included. Thus this is a screening experiment, not the
distributed main controller. It cannot substantiate minimal total resources.

## First stage: corrected moments alone are insufficient

`python -m tools.audit_noncoherent_geometry` tests fixed roles on geometry
seeds1010000..1010003. All8 cases choose the same move under raw/coherent/energy
screening. For example, rho0 episode0 PD .87180 -> .87486; rho.6 .41080 -> .41692.
No novelty advantage is established.

Adding `--role-switch` includes three legal transmitter roles (66 candidates).
All raw/energy choices switch to transmitter1; without clutter this decreases
PD in all four cases. Example episode3/rho.6: stay .55596, coherent-screened
.11656, energy-screened .47660. This rejects automatic promotion of coherent
Schur geometry scoring into the noncoherent receiver.

## Second stage: template-mismatch uncertainty in action screening

After these diagnostics, declare a finite support y in {0,30,60}m and vx in
{0,6,12}m/s. For action a and support state k, compute projected mean-energy
shift delta_(a,k) using the candidate's predicted receive template. Score:

    min_k w_a^T delta_(a,k) / sqrt(w_a^T Sigma_E,H0,a w_a).

Use the signed shift, not its square: a negative detector-direction shift is
not valuable detection evidence. Select the maximum across feasible actions,
including stay. This is a robust H0-normalized shift surrogate, NOT an exact
worst-case PD, e-process, learned belief or optimal information-theoretic policy.
Support includes the evaluation truth, explicitly an in-support test; uncovered
states remain untested. Selection is exhaustive here; no acceleration claim.

Fresh geometry seeds1011000..1011003 and fresh inner streams are used:

`python -m tools.audit_noncoherent_geometry --role-switch --robust-screen --geometry-seed 1011000`

| Geometry | rho | Stay PD | Raw-screen PD | Robust-screen PD | Robust PFA |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | .88716 | .85888 | .89002 | .00034 |
| 1 | 0 | .93678 | .92192 | .93822 | .00060 |
| 2 | 0 | .96188 | .94028 | .96352 | .00052 |
| 3 | 0 | .98542 | .98236 | .98598 | .00040 |
| 0 | .6 | .45736 | .52324 | .52284 | .00044 |
| 1 | .6 | .53554 | .53792 | .54510 | .00040 |
| 2 | .6 | .62328 | .62096 | .62608 | .00042 |
| 3 | .6 | .76222 | .83342 | .83650 | .00050 |

All8 robust-vs-stay point differences are positive; per-case paired intervals
also exclude zero, without multiplicity coverage across cases. Robust does not
dominate raw everywhere (geometry0/rho.6). Three clutter cases remain below
PD>.8; point PFA alone does not certify a family-wide false-alarm requirement.
Robust retains transmitter0 in all independent-noise cases and two clutter
cases, showing that some harmful switches are avoided. No global optimality
or cross-scene guarantee follows.

## Verification and next boundary

The new moment model passes a150k H0 covariance Monte Carlo, phase-invariance
and finite-difference derivative checks. Earlier Schur-gradient regression
remains intact. Both experiment stages run successfully in the PyTorch conda
environment. Online action authority remains unchanged.

Next tests must vary actual target states within and outside the declared
support, H1 phase patterns and estimated covariance, then include actual
message availability and motion costs. Threshold selection must stay separate
from those held-out H1 results. The remaining clutter deficit is not solved by
renaming the score or simply choosing a larger raw SNR.
