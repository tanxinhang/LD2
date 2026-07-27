# Tracking-free MAPPO-ISAC with learned U2U communication

## Current scope

The default experiment now models fixed, mission-known sensing targets. The
Kalman tracking loop is bypassed, target motion is disabled, and there is no
UAV-to-ground reporting link. The only explicit inter-node information path is
cost-aware UAV-to-UAV communication.

The legacy behavior remains selectable through configuration:

- `tracking_enabled=true`: moving targets and belief predict/update;
- `ground_communication_enabled=true`: receiver-to-fusion-center reliability,
  energy, report bits, capacity and latency;
- `learned_comm_mode=on`: historical free continuous messages;
- `learned_comm_mode=off`: no learned messages.

## Learned communication action

For UAV (k), MAPPO samples a joint action

\[
  a_k=(\Delta p_k, r_k, m_k, x_k, y_{k,1:Q}),
\]

where \(m_k\) is unconstrained learned token content,
\(r_k\) is a categorical rate action, \(x_k\) is a continuous communication
power action, and \(y_{k,1:Q}\) is a continuous sensing allocation over all
targets. Rate zero means silence;
the other default rates quantize each message component to 4, 8, 16 or 32
bits. The 32-bit level is optional high precision rather than a raised minimum,
because forcing the minimum from 4 to 8 bits did not improve the 20-seed probe.
The policy is not told what the message must represent.

Both the Gaussian message density and categorical rate probability are included
in the stored PPO log probability. Consequently, future sensing reward and
transport cost can train both message semantics and the decision to transmit.

## Joint communication-sensing power

`config/exp_800_q4_u2u_joint_isac.yaml` enables the formal joint-resource
model.  Each UAV has a fixed RF budget \(P_k^{\rm total}\).  The transformed
power action and rate define

\[
 P_k^{\rm comm}=\mathbf 1[r_k>0]P_k^{\rm total}
 \left(\rho_{\min}+(\rho_{\max}-\rho_{\min})\sigma(x_k)\right),
\]

and the remainder is split across every target,

\[
 P_{kq}^{\rm sense}=\left(P_k^{\rm total}-P_k^{\rm comm}\right)
 \operatorname{softmax}(y_{k,1:Q})_q.
\]

Therefore \(P_k^{\rm comm}+\sum_qP_{kq}^{\rm sense}=P_k^{\rm total}\) to
floating-point precision.  Communication power changes link SNR, Shannon
rate, latency and radio energy; target-wise sensing power directly changes the
bistatic deflection entries used by P0.  A transmitter may illuminate several
targets in the same frame, and P0 may reuse a UAV in several same-role
target/pair assignments while still preventing simultaneous transmitter and
receiver roles.

The resource variables are logistic-normal actions. Their raw Gaussian samples
are stored in the rollout buffer and included in the PPO likelihood ratio, so
the resource heads are trained by sensing utility and transport cost rather
than being deterministic side heads without policy gradients.

### Target-token payload

The formal joint experiment uses `comm_payload_mode=target_tokens`. Rather
than defining fields such as position, target ID, bid or intent, the structured
actor projects each local target-entity hidden state to one 16-D latent token:

\[
 M_k=[\phi(z_{k1}),\ldots,\phi(z_{kQ})]\in\mathbb R^{Q\times16}.
\]

The physical layer flattens, quantizes and charges this payload, then the
receiver restores it as a set of sender-target tokens. Sender ID, target-slot
ID, rate, latency, SNR and AoI are transport metadata, not prescribed message
semantics. Every local target entity queries the complete received token set
through masked cross-attention, so token content and cross-target use remain
learned end to end.

## Learned receiver attention

Delivered messages are kept as separate sender tokens instead of being averaged.
For every possible peer, a token contains the learned 16-dimensional payload and
five transport descriptors (sender identity, rate, latency, SNR and age of
information), together with a delivery mask. Each sensing-target embedding acts
as a query in masked cross-attention over these tokens:

\[
  z_q' = \operatorname{LN}\!\left(z_q +
  g_q\odot\operatorname{MHA}(z_q, M, M;\mathcal M)\right).
\]

Thus the sender learns what and when to transmit, while the receiver learns
which sender message is useful for each target. Missing, expired or undelivered
messages are masked and cannot influence the action. A delivered token is kept
for a configurable number of frames with increasing age, so the policy may
trade fresh communication against reuse of older information.

## Transport and cost

An active sender broadcasts one quantized payload:

\[
  L_k = H + 16 b_{\rho_k}\quad\text{bits}.
\]

Active senders share the U2U bandwidth orthogonally. Each receiver link uses a
Friis path gain and Shannon rate. A message is exposed only to its intended
receiver inbox when its SNR and deadline constraints pass. The environment
records payload bits, serialization/processing latency, delivery rate and radio
energy (E_k=P_{\rm U2U}t_k).

The sensing team reward is reduced by

\[
  c_{\rm comm}=\lambda_b L+\lambda_e E+
  \lambda_d(\bar t+v_{\rm deadline}).
\]

Default weights and link parameters are in `config/default.yaml`.

When `comm_qos_constrained=true`, training treats communication as a
minimum-resource problem subject to sensing floors rather than relying only on
a fixed weighted sum. Three non-negative dual variables correspond to steady,
weak-3 and worst-target detection. A multiplier increases when its rollout
metric is below the configured floor and decreases after the floor is met. The
explicit bit/energy/delay cost then favours silence and lower rates among
quality-feasible policies. The default floors are 0.80, 0.70 and 0.60.

Because the shared QoS return is difficult to attribute to a sender's discrete
transmit/silent choice, successful active senders receive a small participation
bonus. Its strength grows with the largest normalized QoS deficit, and falls to
`comm_encouragement_floor_ratio` of its configured value once all floors are
met. Failed packets and silence receive no bonus. The bonus is identical for
4/8/16/32-bit packets: it opens the communication channel for exploration but
does not reward high precision. Bit, energy and delay costs remain active, so
the trained policy must still prune useless transmissions and choose the lowest
rate that preserves sensing performance.

The older `comm_qos_min_rate_bits` feasibility barrier remains available for
ablation but is disabled by default. Forcing every UAV to at least 8 bits made
all four UAVs transmit and raised traffic to 768 bits per decision, without a
corresponding sensing gain in the fixed-seed probe.

The current configuration instead uses a QoS-gated soft rate-exploration
bonus. Precision credit rises from 4 to 8 bit/dimension and saturates at 8 bit.
Thus 16/32 bit receive no larger bonus and must justify their additional bit,
energy and latency costs. This increases communication volume without imposing
an environment-side minimum. Training and evaluation log the full rate
distribution and mean selected bits per message dimension. Because the pure
PPO rate signal was too weak to change deterministic deployment decisions, an
auxiliary rate-head-only optimizer softly pulls the categorical head toward 8
bit while QoS is infeasible; it deactivates as soon as all three floors are met.

Short deliberate silence remains free, but persistent silence is now treated
as communication-channel abandonment. By default, each sender may remain
silent for three consecutive policy decisions (15 simulator frames in the
Medium configuration). Every
additional silent decision incurs 0.01 reward penalty up to 0.05; any non-zero
rate resets that sender's streak. This liveness penalty does not prescribe
message content or payload precision.

Checkpoint selection is lexicographic: full QoS feasibility first, then
worst-target P_D, weak-3 P_D, steady P_D, and finally lower communication bits.
Thus traffic and steady mean can only break ties; neither can displace a policy
with better worst-target protection.

The joint experiment uses `actor_decision_interval=1`: every physical frame is
one decentralized communication round.  The message sent in round \(t+1\) is
computed from an observation that can include messages delivered after round
\(t\), which supports repeated learned bids, acknowledgements or consensus
without prescribing those semantics.  Every round is separately charged for
bits, airtime, delay and energy.  The legacy macro-action configurations still
send only one packet at the beginning of each held movement decision.

The target-token experiment additionally sets
`movement_decision_interval=2`. Communication content, rate, communication
power and target-wise sensing power are resampled every frame, while only the
movement/role action is locally held for two frames. Consequently, the second
token exchange observes the first exchange's delivered inbox before the next
movement commitment. PPO stores a movement-head mask: held frames contribute
communication/resource log probability and entropy but no spurious movement
log probability. This changes no execution information boundary and requires
no centralized scheduler.

An experimental `round_negotiation_enabled` branch adds a synchronized local
proposal/response phase and the UAV's own node identifier to this path. The
response token is generated after received proposal tokens have updated the
target entities. A shared learned claim projection reads both the UAV's own
outgoing latent target token and received quantized target tokens, providing a
comparable local peer-exclusion score. Neither the environment nor the loss
assigns values or named fields to the 16-D token. Team grouping is used only
during CTDE for an unlabeled coverage/exclusivity regularizer; deployment uses
the local observation, local clock, node ID and physical inbox only. The probe
is configured separately in
`config/exp_800_q4_u2u_round_agreement.yaml` and is not the deployable default.

## Information boundary

With `comm_only_neighbor_information=true`, the historical neighbor
position/velocity/role/intent slots are zero-filled. Their dimensions are kept
for checkpoint and parser compatibility, but learned U2U messages become the
only inter-UAV information path. The legacy averaged-message slot is also
zero-filled when cross-attention is enabled, preventing a bypass around the
per-sender delivery mask. Each UAV still observes its own state and the
mission-known target coordinates.

## Logged metrics

Training and evaluation now expose:

- bits, radio energy and active senders per simulator frame;
- mean latency and delivery rate over actual packet attempts;
- active senders and learned silence rate;
- the existing steady, weak-3 and worst-target sensing metrics.
- communication/sensing power per frame, per-target sensing power and maximum
  per-UAV power-balance error.

For publication experiments, compare at least `off`, legacy free `on`, and
`cost_aware` under identical seeds, and report sensing quality jointly with the
communication Pareto metrics rather than tuning only for detection probability.

## Experimental allocation modules

The code contains disabled CTDE assignment-teacher, differentiable quantized
inbox and Sinkhorn team-consistency modules. These preserve learned 16-D message
contents and add no privileged field to the deployed actor observation. They
are research probes, not the current default: offline assignment exceeded
99.9%, but physical recurrent-inbox evaluation degraded sensing. Use
`config/exp_800_q4_u2u_teacher.yaml` only to reproduce this negative result; do
not use its checkpoints as the paper's main method.
