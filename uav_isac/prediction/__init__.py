"""Constraint-native predictive structure, power and detection components."""

from .certified_gnn import (
    append_causal_feature_residual,
    audit_predictive_checkpoint,
    CertifiedBipartiteGNN,
    FactorizedPrediction,
    PrefetchedPrediction,
    PredictionPrefetchCache,
    PREDICTIVE_CHECKPOINT_SCHEMA,
    build_endpoint_features,
    augment_temporal_protocol_features,
    decode_candidate_pool,
    require_compatible_predictive_checkpoint,
    validate_selected_prediction,
)
from .refresh_gate import PredictiveRefreshGate, RefreshDecision, RefreshPath
from .constrained_objective import (
    ConstrainedObjectiveResult,
    constrained_detection_objective,
    detection_probability_from_deflection,
    project_power_to_row_budget,
)
from .gradient_surgery import physical_anchor_pcgrad

__all__ = [
    "append_causal_feature_residual",
    "audit_predictive_checkpoint",
    "CertifiedBipartiteGNN",
    "FactorizedPrediction",
    "PrefetchedPrediction",
    "PredictionPrefetchCache",
    "PREDICTIVE_CHECKPOINT_SCHEMA",
    "build_endpoint_features",
    "augment_temporal_protocol_features",
    "decode_candidate_pool",
    "require_compatible_predictive_checkpoint",
    "validate_selected_prediction",
    "PredictiveRefreshGate",
    "RefreshDecision",
    "RefreshPath",
    "ConstrainedObjectiveResult",
    "constrained_detection_objective",
    "detection_probability_from_deflection",
    "project_power_to_row_budget",
    "physical_anchor_pcgrad",
]
