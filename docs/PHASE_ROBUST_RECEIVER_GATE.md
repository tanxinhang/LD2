# P1: frozen-schedule scalar phase diagnostic

Eight seeds and 100000 samples in each independent calibration/H0/H1 split:

| Receiver | Mean P_D |
| --- | ---: |
| Prediction complex coherent | 0.230355 |
| Prediction scalar energy GLRT | 0.213670 |
| Truth complex coherent reference | 0.995735 |
| Truth scalar energy reference | 0.981546 |

Held-out P_FA is 0.00117 (coherent) and 0.00104 (energy). Neither is a
certification of P_FA <= 0.001. All decisions occur at the terminal fifth
epoch. This is centralized with known proper Gaussian covariance. The new
complex coherent receiver is different from the earlier real-projection
receiver, so its 0.230355 must not be advertised as a phase-only improvement
over the old 0.153731 result.

The scalar energy test removes a common phase rotation of the final complex
statistic. It cannot undo cancellation between acquisitions before that
statistic was formed, or restore signal orthogonal to the predicted template.
Thus this experiment does not yet validate independent-phase likelihood fusion.

## Corrections to the supplied proposal

1. Conditional Gaussian distributions generally have nonzero means even when
   the joint distribution is zero mean. With history h, m_k=C_k,aH C_k,HH^-1 h.
   Proper complex conditional KL includes (m_1-m_0)^H V_0^-1(m_1-m_0), as well
   as tr(V_0^-1 V_1)-d+log(det(V_0)/det(V_1)). Dropping the mean term is wrong.
2. A Gaussian complex amplitude prior changes both amplitude and phase. It is
   not equivalent to fixed amplitude with uniform unknown phase. The rank-one
   KL lambda-log(1+lambda) applies to that Gaussian prior specifically.
3. Sum pi_l KL(p_l || p_0) upper-bounds KL(sum pi_l p_l || p_0); generally they
   are unequal. The gap measures state information in the observation under
   the mixture model. Expected state-specific information requires a surrogate
   label until the actual mixture receiver is validated.
4. Ninety-nine percent retained SVD energy does not guarantee preserved rare
   state detection or false alarm behavior. Probability masses must survive any
   compression, and the final detector must be independently recalibrated.

The next diagnostic should distinguish common amplitude/phase from independent
per-acquisition amplitudes, then evaluate a belief-derived mixture with a
terminal global threshold. Geometry/power schedules remain frozen for that
comparison. No P_D > 0.8 or distributed-system success is claimed here.

Reproduce: `python tools/audit_phase_robust_receiver.py`.
