# D095 — Joint L2 (structure) × L3 (geometry): alternating block descent

## 1. The correction that changed the plan

The D0.94 "joint 71.7%" was a **relaxed upper bound** that let one UAV be both
transmitter and receiver.  The deployed architecture is half-duplex
(`full_duplex=False`): a UAV is TX *or* RX in a frame.  Re-measuring with the
single-role oracle on hard frames (γ = ∞):

| lever | hard-frame repair (γ ≤ 1) |
|---|---|
| L2 structure oracle (single-role, 2⁸ enumeration) | 26.7% |
| L3 geometry (fixed teacher structure) | 20–47% |
| one-shot joint (oracle structure → L3) | 60% |
| **alternating joint (greedy structure ↔ L3)** | **67.5%** |

The honest joint ceiling is ~60–67%, not 72%: the half-duplex role partition is
a real constraint (it costs ~12 pts off the relaxed estimate).

## 2. Why structure is a major (not minor) lever

Gain `a_ijq ∝ 1/(R_tx²·R_rx²)`.  The owner j is in the denominator of *every*
transmitter's gain to target q, so a far owner suppresses all TXs at once.
Reassigning the owner to a nearer receiver is a **zero-movement-cost lever** that
lifts all TX gains by `(R_old/R_new)²`.  Measured structure-alone headroom on
hard frames (worst floor, power sharing included):

| structure method | recoverable |
|---|---|
| relaxed best-owner (full-duplex, no role) | 52% (upper bound) |
| priced greedy (half-duplex, power-sharing-aware LP score) | 25.7% |
| oracle (half-duplex, 2⁸ partition MILP) | ~40% (est.) |

## 3. The innovation: price-mediated structure repair

`uav_isac/coordination/priced_structure.py`.  The envelope theorem
`∂γ*/∂a_iq = −π_q·p_iq` gives one price vector π that drives **both** the
geometry gradient (advice 008 T2) and the structure choice.  Owner selection is
price-orthogonal (`j_q* = argmax_j Σ_i a_ijq b_i`, π_q factors out); the
half-duplex role partition is resolved by a greedy that scores every candidate
RX set with the **exact max-min power LP** (power-sharing aware, not the
optimistic ceiling).  This replaces the oracle's 2⁸ enumeration with a
price-justified greedy and keeps the distributed info boundary (each UAV
proposes its own marginal `Σ_q π_q a_kjq`).

## 4. The alternating descent (the payoff)

Block coordinate descent on `min_{x,S} γ*(x,S)`:

    repeat
      L2: S ← argmin_S γ*(x, S)         (priced greedy repair)
      L3: x ← argmin_x γ*(x, S)         (bounded deficit→capability descent)
      rescale full (K,K,Q) tensor by Friis 1/(R_tx²R_rx²) after the move
    until γ ≤ 1 or max iters

`tools/audit_l3_l2_alternating.py` (full-tensor Friis rescale verified against
the fixed-owner rescale).  Result on 40 hard frames, 4 iters × 25 steps:

- L3 alone: **47.5%**
- alternating joint: **67.5%**  (+20 pts)

Alternation with the *weak* greedy per-step solver beats the *strong* one-shot
oracle (60%): re-optimising the structure after geometry moves recovers more
than a better one-shot structure.

## 5. End-to-end integration and the price that matches the metric

`analytical_movement_enabled` in `env_core.step()` overrides the learned
trajectory with one step of the deficit→capability descent each frame
(`_analytical_movement_delta`).  Structure (L2) is re-optimised by P0 at the
moved geometry; power (L1) and comm (L0) are the task-constrained analytical
stack.

Key theoretical point — *the metric selects the price*.  The evaluation's
`steady` is the temporal mean of the **worst** target's P_D over the last 20
frames, so steady is a monotone function of the worst target alone.  The
max-min dual `λ*` concentrates on exactly the bottleneck (worst) target by
complementary slackness, whereas the full capability-gauge price `π` spreads
over worst + bottom-3 + steady and *dilutes* the motion.  Therefore the
geometry layer should be driven by `λ*`, and the power layer by the full gauge.
This is a clean separation of concerns:

- L1 power: capability gauge (all three floors) — `task_constrained_power_enabled`
- L3 motion: max-min dual price (worst floor = what `steady` measures) — cheap LP, no PWL

3-seed end-to-end (max-min-price hook, hover-gated):

| metric | task-constrained baseline | + L3 motion |
|---|---|---|
| worst | 0.609 | **0.730** |
| weak3 | — | **0.774** |
| steady | 0.736 | **0.841** |
| QoS feasible | 0.50 | **0.67** |

Two correctness fixes were required for a *monotone* (never-worse) hook:

1. **Gate on the steady floor.**  The max-min price should drive motion only
   while the worst target is below the steady floor (0.80); otherwise a
   seed that was already perfect (worst 1.0) was degraded to 0.48 by moving the
   UAVs away from a good geometry.
2. **Hover, don't hand back to the actor.**  When the floor is met the hook must
   return a zero displacement (hover), not fall through to the actor's motion,
   which oscillates and re-degrades a hard seed (seed 34 stalled at 0.61 under
   fallback vs 0.75 under a persistent gradient; hover keeps it at the floor).

The hook is therefore a *satisficing* controller: drive the worst target to the
steady floor, then hold.  **20-seed confirmation** (task-constrained baseline →
+ L3 motion):

| metric | baseline | + L3 motion | floor |
|---|---|---|---|
| worst (mean) | 0.609 | **0.662** | ≥0.60 |
| weak3 (mean) | — | **0.724** | ≥0.70 |
| steady (mean) | 0.736 | **0.808** | ≥0.80 |
| QoS feasible | 0.50 | **0.65** | ≥0.70 |

The steady floor is met on **all 20 seeds** (0/20 below 0.80); the worst floor
is met on the mean (7/20 seeds still dip below 0.60 in the *early* frames,
before the motion converges — a known receding-horizon transient).

## 6. Status and next step

Payoff confirmed end-to-end (worst 0.681 / weak3 0.725 / steady 0.795, near the
0.60/0.70/0.80 floors on 3 seeds).  Next: confirm on 20 seeds; then close the
remaining ~0.005 steady gap and push the QoS-feasible rate.
