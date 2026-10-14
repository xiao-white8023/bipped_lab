"""Directed collection provenance. These tensors never enter selector features."""

import torch


SOURCE_MODES = ("any", "lp_only", "vp_only", "dual")
SOURCE_AUDIT_ORDER = (
    "lp_effect_in_history", "vp_effect_in_history",
    "lp_clean_label_window", "vp_clean_label_window",
    "lp_effect_age_s", "vp_effect_age_s", "lp_severity", "vp_severity",
)
VP_AUDIT_ORDER = (
    "lp_strength", "vp_strength", "local_bias_m", "local_scale_error",
    "edge_mix", "warp_shift_x_px", "warp_shift_y_px", "warp_rotation_deg",
    "mask_fraction", "mask_speed_px_per_s", "missing_return_mode",
    "filter_alpha", "filter_error_max_m", "actual_delay_s", "holding_frame",
    "current_effect", "current_support_ratio", "current_saturation_ratio",
    "new_capture", "history_append", "source_age_s",
    "mask_center_0_x", "mask_center_0_y", "mask_center_1_x", "mask_center_1_y",
    "mask_center_2_x", "mask_center_2_y", "mask_direction_x", "mask_direction_y",
    "field_mean", "field_std", "changed_pixel_fraction",
    "history_interval_s",
)


def directed_audit_contract():
    return {
        "version": 1, "source_order": list(SOURCE_AUDIT_ORDER),
        "vp_order": list(VP_AUDIT_ORDER),
        "source_modes": list(SOURCE_MODES),
        "history_frames": 5, "horizon_s": 1.0,
        "effect_age_missing": -1.0,
        "source_age_missing": -1.0,
        "clean_label_window": "no injected fault in observable history or label window; current data available",
        "missing_return_mode": ["zero_no_return", "nonfinite_no_return"],
        "severity": "per branch; legacy plan severity is the maximum for independent dual events",
        "scope": "audit and admission only; excluded from all selector features",
        "vp_model": "input-level surrogate on captured depth in metres; not a camera renderer",
        "mask_centers": "normalized image coordinates at parent start; motion uses direction, speed and event age",
        "recovery_effect": "includes corrupted images remaining in VP/selector history; fully clean history is ineligible",
        "history_interval_s": "newer delivered source timestamp minus previous; can be negative for reordered arrivals",
    }


def source_rejection_masks(mode, faults, source):
    """Check source AND observed effects; labels are checked separately."""
    if mode not in SOURCE_MODES:
        raise ValueError(f"unknown fault source mode: {mode}")
    n = len(faults)
    if faults.shape != (n, 8):
        raise ValueError("fault provenance requires [n,8] fault metadata")
    zero = torch.zeros(n, device=faults.device, dtype=torch.bool)
    if mode == "any":
        return {"wrong_source": zero, "effect_not_observable": zero, "healthy_side_not_clean": zero}
    if source is None or source.shape != (n, len(SOURCE_AUDIT_ORDER)) or not torch.isfinite(source).all():
        raise ValueError("directed collection requires finite source audit metadata")
    if not ((source[:, :4] == 0) | (source[:, :4] == 1)).all():
        raise ValueError("source effect/clean flags must be binary")
    lp, vp = faults[:, 0].ne(0), faults[:, 1].ne(0)
    if mode == "lp_only":
        match, effect, clean = lp & ~vp, source[:, 0].bool(), source[:, 3].bool()
    elif mode == "vp_only":
        match, effect, clean = ~lp & vp, source[:, 1].bool(), source[:, 2].bool()
    else:
        match = lp & vp
        effect = source[:, 0].bool() & source[:, 1].bool()
        clean = torch.ones_like(match)
    # Exclusive rejection reasons, in the order admission evaluates them.
    return {"wrong_source": ~match, "effect_not_observable": match & ~effect,
            "healthy_side_not_clean": match & effect & ~clean}
