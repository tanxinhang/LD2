# Sender-token causal-value feasibility probe

## Question

Before adding a sequential communication gate to the deployed actor, test
whether the finite-horizon marginal QoS value of one sender-target token can be
predicted from strictly decentralized information.

The probe leaves the frozen policy unchanged. At one policy state it deep-copies
the complete simulator, sends or masks exactly one active token, and advances
both branches from the same simulator, actor-memory, movement-hold, and random
state. All later actions are deterministic but may react to the intervened
message. Training and testing use disjoint episode seeds.

## Protocol

- Frozen checkpoint:
  `results/qos_pretrained_soft_gated_move50_test100/best_restored.pt`
- Scenario: 4 UAVs / 4 targets, target-token U2U communication.
- Training seeds: 634, 754, 802, 325.
- Confirmation seeds: 119, 871, 702, 575.
- Sampling: every 5 frames, 30 interventions per episode.
- Rows: 120 train and 120 confirmation.
- Intervention: one sender-target mask, send versus drop.
- Mean marginal communication load: 128 bit per intervention.
- QoS value:
  `2 * feasible + worst + 0.25 * weak3 + 0.10 * steady`.
- Medium feasibility thresholds: steady 0.80, weak3 0.70, worst 0.60.
- Predeclared integration gates: local AUC >= 0.65, local Spearman >= 0.20,
  and recovered hindsight-oracle gain >= 0.50.

Exact replay control repeated the same five-frame send branch twice and gave a
maximum absolute metric error of 0.0.

## Results

### Two-frame direct-effect horizon

| Feature view | ROC AUC | Spearman | Recovered oracle gain |
|---|---:|---:|---:|
| Protocol only | 0.566 | 0.228 | 0.719 |
| Raw decentralized observation | 0.510 | 0.159 | 0.224 |
| Actor decentralized latent state | 0.587 | 0.094 | approximately 0 |
| Privileged centralized view | 0.585 | -0.055 | 0.029 |

The confirmation marginal-QoS distribution is broad: mean -0.0398, standard
deviation 0.4257, with 44.2% positive, 28.3% negative, and 27.5% neutral rows.
Always sending the sampled token loses 0.0398 QoS value relative to always
dropping it, while the per-row hindsight oracle gain is 0.0654.

An additional model-family check did not reverse the conclusion. Regularized
ridge/PCA, logistic regression, and ExtraTrees all failed to give the richer
local or centralized views a stable continuous ranking advantage. Protocol-only
ExtraTrees classification reached AUC 0.647, but this is just below the gate and
did not establish a calibrated marginal-value estimator.

### Five-frame closed-loop horizon

| Feature view | ROC AUC | Spearman | Recovered oracle gain |
|---|---:|---:|---:|
| Protocol only | 0.590 | 0.092 | -0.292 |
| Raw decentralized observation | 0.503 | -0.094 | -1.247 |
| Privileged centralized view | 0.582 | -0.355 | -1.522 |

The longer horizon amplifies hard matching, communication, and movement
feedback. Confirmation marginal QoS has mean -0.0844 and standard deviation
0.3909. Per-seed means range from -0.4438 to +0.0942, so a small set of states
does not define a transferable decision boundary.

## Interpretation

The simulator contains genuine token-level causal effects, but the current
finite-horizon scalar value is not yet predictably identifiable from local
execution information on unseen episodes. A richer input does not solve the
problem; with this dataset it mostly increases estimation variance. The weak
protocol-only result is consistent with scenario-specific rank, phase, sender,
or target regularities and is not evidence of general causal adaptation.

The five-frame result also shows why a direct `send if predicted value > cost`
gate is unsafe today. A small message intervention crosses hard assignment and
QoS-feasibility boundaries, producing a discontinuous long-term label. A value
head trained on this target could remove critical messages on new geometries.

## Decision

Do not integrate the sequential causal-value gate into MAPPO yet. Keep the
probe as an offline diagnostic and leave the deployed policy unchanged.

The next defensible experiment is to replace the single discontinuous team
label with a multi-head, target-wise short-horizon teacher: predict the token's
effect on each target P_D and on the worst-target identity, estimate uncertainty,
and form a conservative lower confidence bound only after prediction. It should
be trained on a much broader seed/domain bank and must beat protocol-only models
on unseen geometries before controlling communication.

## Final target-wise follow-up

The proposed target-wise teacher was implemented and tested on the same H=2,
120/120 episode-disjoint dataset. Training-time gate selection used leave-one-
episode-out predictions, and tree-ensemble disagreement supplied the uncertainty
term. The safety constraint required at least 0.75 recall on tokens that improve
the no-token branch's worst target.

| Local target-wise metric | Result | Required |
|---|---:|---:|
| Per-target effect sign AUC | 0.728 | >= 0.65 |
| Per-target effect Spearman | 0.468 | >= 0.20 |
| Worst-target top-2 accuracy | 0.642 | >= 0.70 |
| Critical-repair recall | 1.000 | >= 0.75 |
| Recovered oracle gain | -0.608 | >= 0.50 |
| Confirmation gate gain | -0.0398 | > 0 |

Decomposing the label therefore solved part of the identifiability problem:
token effects on individual targets are rankable on unseen episodes. However,
the sender still cannot identify the team's current worst target reliably. To
preserve critical-repair recall, the selected conservative gate activates on
100% of confirmation rows, exactly reproducing the negative always-send gain.

This is the stopping result for the causal-value gate route. The target-wise
predictor remains an offline diagnostic, but neither it nor the gate is connected
to MAPPO execution. Further threshold or architecture tuning on these seeds
would contradict the stated generalization objective.
