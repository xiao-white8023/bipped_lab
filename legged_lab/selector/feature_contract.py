"""Semantic version of observable features and counterfactual execution."""

from copy import deepcopy
from typing import Any


def selector_feature_contract() -> dict[str, Any]:
    return {
        "version": 3,
        "feature_shape": [5, 283],
        "sample_period_s": 0.1,
        "candidate_boundary": "last_feature_snapshot_and_label_start_same_control_boundary",
        "health": "raw_valid_support;packet_cached_temporal;float64_source_and_packet_time",
        "mode": "accepted_execution_target;switch_time_before_first_transition_action",
        "transition": {"normal_steps": 5, "emergency_steps": 2, "control_dt_s": 0.02},
        "counterfactual": "hold_requested_branch_for_1s_with_snapshotted_action_blender",
        "randomness": "row_local_named_streams_v1",
        "fault_geometry": "physical_xy_metres;bilinear_validity_weighted_height_and_support_v1",
        "fault_ranges": "exact_configured_bounds_including_zero_v1",
    }


def validate_feature_contract(value: Any) -> dict[str, Any]:
    if value != selector_feature_contract():
        raise ValueError(
            "selector feature contract is missing/outdated; recollect and retrain"
        )
    return deepcopy(value)


def dataset_feature_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("metadata", {}).get("collection_complete") is not True:
        raise ValueError(
            "selector collection is incomplete; do not train on a partial dataset"
        )
    return validate_feature_contract(
        manifest.get("metadata", {}).get("selector_feature_contract")
    )

