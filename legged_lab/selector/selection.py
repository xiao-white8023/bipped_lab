"""One-second danger-to-safe requests, shared by offline and online code.

This module only proposes a target. The runtime still applies three
confirmations, minimum dwell and the ordinary action blend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class SelectionRule:
    danger_threshold: float = 0.50
    safe_threshold: float = 0.10
    mode: str = "danger_to_safe_only"
    comparison: str = "probability"
    both_safe_hold_risk: float = 0.001
    both_unreliable_risk: float = 0.99
    lp_to_vp_min_current_risk: float = 0.0
    vp_to_lp_min_current_risk: float = 0.0
    lp_to_vp_max_alternative_risk: float = 1.0
    vp_to_lp_max_alternative_risk: float = 1.0
    lp_to_vp_margin: float = 0.0
    vp_to_lp_margin: float = 0.0
    reference_scale: tuple[float, float] = (1.0, 1.0)
    reference_bias: tuple[float, float] = (0.0, 0.0)
    both_safe_probability_threshold: float = 0.99
    both_failure_flag_threshold: float = 0.20
    both_failure_hold_threshold: float = 0.60
    lp_to_vp_min_rescue_probability: float = 0.01
    vp_to_lp_min_rescue_probability: float = 0.01

    def validate(self) -> None:
        if self.mode == "learned_outcomes":
            for name in ("both_safe_probability_threshold", "both_failure_flag_threshold",
                         "both_failure_hold_threshold", "lp_to_vp_min_rescue_probability",
                         "vp_to_lp_min_rescue_probability"):
                value = getattr(self, name)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name} must lie in [0, 1]")
            if self.both_failure_hold_threshold < self.both_failure_flag_threshold:
                raise ValueError("joint-failure hold threshold must be at least the warning threshold")
            margins = (self.lp_to_vp_margin, self.vp_to_lp_margin)
            if not all(math.isfinite(x) for x in margins) or sum(margins) < 0:
                raise ValueError("directional margins must be finite and non-overlapping")
            return
        if self.mode == "directional_comparison":
            if self.comparison not in ("probability", "log_odds", "reference_score"):
                raise ValueError("unsupported directional comparison")
            for name in ("both_safe_hold_risk", "both_unreliable_risk",
                         "lp_to_vp_min_current_risk", "vp_to_lp_min_current_risk",
                         "lp_to_vp_max_alternative_risk", "vp_to_lp_max_alternative_risk"):
                value = getattr(self, name)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name} must lie in [0, 1]")
            if self.both_safe_hold_risk >= self.both_unreliable_risk:
                raise ValueError("low-risk hold boundary must be below both-unreliable boundary")
            margins = (self.lp_to_vp_margin, self.vp_to_lp_margin)
            if not all(math.isfinite(x) for x in margins) or sum(margins) < 0:
                raise ValueError("directional margins must be finite and non-overlapping")
            if (len(self.reference_scale) != 2 or len(self.reference_bias) != 2
                    or not all(math.isfinite(x) and x > 0 for x in self.reference_scale)
                    or not all(math.isfinite(x) for x in self.reference_bias)):
                raise ValueError("reference mapping needs two positive scales and finite biases")
            return
        if self.mode != "danger_to_safe_only":
            raise ValueError("unsupported selection mode")
        if (not math.isfinite(self.danger_threshold)
                or not math.isfinite(self.safe_threshold)
                or not 0 <= self.safe_threshold < self.danger_threshold <= 1):
            raise ValueError("selection thresholds must satisfy 0 <= safe < danger <= 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        if self.mode == "learned_outcomes":
            names = ("both_safe_probability_threshold", "both_failure_flag_threshold",
                     "both_failure_hold_threshold", "lp_to_vp_min_rescue_probability",
                     "vp_to_lp_min_rescue_probability", "lp_to_vp_margin", "vp_to_lp_margin")
            return {"version": 3, "mode": self.mode, **{name: getattr(self, name) for name in names}}
        if self.mode == "directional_comparison":
            names = ("comparison", "both_safe_hold_risk", "both_unreliable_risk",
                     "lp_to_vp_min_current_risk", "vp_to_lp_min_current_risk",
                     "lp_to_vp_max_alternative_risk", "vp_to_lp_max_alternative_risk",
                     "lp_to_vp_margin", "vp_to_lp_margin", "reference_scale", "reference_bias")
            return {"version": 2, "mode": self.mode,
                    **{name: list(getattr(self, name)) if name.startswith("reference_")
                       else getattr(self, name) for name in names}}
        return {"version": 1, "mode": "danger_to_safe_only",
                "danger_threshold": self.danger_threshold,
                "safe_threshold": self.safe_threshold}

    @classmethod
    def from_dict(cls, value: Any) -> "SelectionRule":
        if isinstance(value, dict) and value.get("version") == 3 and value.get("mode") == "learned_outcomes":
            fields = dict(value)
            fields.pop("version")
            rule = cls(**fields)
            rule.validate()
            return rule
        if isinstance(value, dict) and value.get("version") == 2 and value.get("mode") == "directional_comparison":
            fields = dict(value)
            fields.pop("version")
            for name in ("reference_scale", "reference_bias"):
                fields[name] = tuple(float(x) for x in fields[name])
            rule = cls(**fields)
            rule.validate()
            return rule
        if (not isinstance(value, dict) or value.get("version") != 1
                or value.get("mode") != "danger_to_safe_only"):
            raise ValueError("selection rule must use danger_to_safe_only version 1")
        rule = cls(float(value["danger_threshold"]), float(value["safe_threshold"]))
        rule.validate()
        return rule


def current_mask_from_features(features: torch.Tensor) -> torch.Tensor:
    """Read the observable execution mask, before feature normalization."""
    if features.ndim != 3 or features.shape[1:] != (5, 283):
        raise ValueError("selector features must have shape [batch, 5, 283]")
    mask = features[:, -1, -2].float()
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("current branch feature must be binary: LP=1, VP=0")
    return mask


@torch.no_grad()
def propose_selection(
    risks: torch.Tensor,
    current_mask: torch.Tensor,
    rule: SelectionRule,
    *,
    lp_available: torch.Tensor | None = None,
    vp_available: torch.Tensor | None = None,
    outcome_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold unless the current branch is dangerous and the other is safe.

An unavailable sensor can make its current branch dangerous. It never makes
the alternative safe: that branch still needs low risk AND availability.
Uncertain intermediate risks hold. Both dangerous branches hold and flag.
"""
    rule.validate()
    if risks.ndim != 2 or risks.shape[1] != 2:
        raise ValueError("risks must have shape [batch, 2]")
    if not torch.isfinite(risks).all() or not ((risks >= 0) & (risks <= 1)).all():
        raise ValueError("risks must be finite probabilities in [0, 1]")
    mask = current_mask.to(device=risks.device).reshape(-1)
    if mask.shape != (risks.shape[0],) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("current_mask must contain one binary mask per sample")
    available = []
    for value in (lp_available, vp_available):
        if value is None:
            available.append(torch.ones_like(mask, dtype=torch.bool))
        else:
            value = value.to(device=risks.device).reshape(-1).bool()
            if value.shape != mask.shape:
                raise ValueError("sensor availability must contain one value per sample")
            available.append(value)
    lp_ok, vp_ok = available
    if rule.mode == "learned_outcomes":
        from .outcome_decision import propose_outcome_selection
        if outcome_probabilities is None:
            raise ValueError("learned outcome decisions require four outcome probabilities")
        return propose_outcome_selection(risks, outcome_probabilities, mask, rule, lp_ok, vp_ok)
    if rule.mode == "directional_comparison":
        score = preference_scores(risks, rule)
        gap = score[:, 0] - score[:, 1]
        low_hold = (risks <= rule.both_safe_hold_risk).all(dim=1)
        both_high = (risks >= rule.both_unreliable_risk).all(dim=1)
        desired = mask.clone()
        eligible = lp_ok & vp_ok & ~low_hold & ~both_high
        to_vp = ((mask == 1) & eligible & (risks[:, 0] >= rule.lp_to_vp_min_current_risk)
                 & (risks[:, 1] <= rule.lp_to_vp_max_alternative_risk)
                 & (gap > rule.lp_to_vp_margin))
        to_lp = ((mask == 0) & eligible & (risks[:, 1] >= rule.vp_to_lp_min_current_risk)
                 & (risks[:, 0] <= rule.vp_to_lp_max_alternative_risk)
                 & (-gap > rule.vp_to_lp_margin))
        # Health alone never nominates a high-risk alternative.
        to_vp |= (mask == 1) & ~lp_ok & vp_ok & (risks[:, 1] <= min(.1, rule.lp_to_vp_max_alternative_risk))
        to_lp |= (mask == 0) & ~vp_ok & lp_ok & (risks[:, 0] <= min(.1, rule.vp_to_lp_max_alternative_risk))
        desired[to_vp] = 0
        desired[to_lp] = 1
        both_flag = both_high | (~lp_ok & ~vp_ok)
        both_flag |= ~lp_ok & (risks[:, 1] > .1)
        both_flag |= ~vp_ok & (risks[:, 0] > .1)
        return desired, both_flag
    lp_danger = (risks[:, 0] >= rule.danger_threshold) | ~lp_ok
    vp_danger = (risks[:, 1] >= rule.danger_threshold) | ~vp_ok
    lp_safe = (risks[:, 0] <= rule.safe_threshold) & lp_ok
    vp_safe = (risks[:, 1] <= rule.safe_threshold) & vp_ok
    desired = mask.clone()
    desired[(mask == 1) & lp_danger & vp_safe] = 0
    desired[(mask == 0) & vp_danger & lp_safe] = 1
    return desired, lp_danger & vp_danger


def preference_scores(risks: torch.Tensor, rule: SelectionRule) -> torch.Tensor:
    """Comparison scores are distinct from calibrated failure probabilities."""
    if rule.comparison == "probability":
        return risks
    logits = torch.logit(risks.float().clamp(1e-12, 1-1e-7))
    if rule.comparison == "log_odds":
        return logits
    scale = risks.new_tensor(rule.reference_scale)
    bias = risks.new_tensor(rule.reference_bias)
    return torch.sigmoid(logits * scale + bias)


@torch.no_grad()
def fit_selection_rule(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    current_mask: torch.Tensor,
    *,
    baseline_probabilities: torch.Tensor | None = None,
    reference_scale: tuple[float, float] = (1.0, 1.0),
    reference_bias: tuple[float, float] = (0.0, 0.0),
) -> tuple[SelectionRule, dict[str, Any]]:
    from .decision_tuning import fit_directional_rule
    return fit_directional_rule(probabilities, labels, current_mask,
        baseline_probabilities=baseline_probabilities,
        reference_scale=reference_scale, reference_bias=reference_bias)


@torch.no_grad()
def fit_absolute_selection_rule(
    probabilities: torch.Tensor, labels: torch.Tensor, current_mask: torch.Tensor,
) -> tuple[SelectionRule, dict[str, Any]]:
    """Choose thresholds on validation ONLY, never on test rows.

Minimize unsafe choices among single-failure samples, then the worse of the
two directional error rates, then switches from an actually safe branch.
No threshold or gate is relaxed based on a test result.
"""
    if probabilities.shape != labels.shape or probabilities.shape[1:] != (2,):
        raise ValueError("validation probabilities and labels must have shape [samples, 2]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("validation labels must be binary")
    masks = current_mask.to(probabilities.device).reshape(-1)
    labels = labels.to(probabilities.device).bool()
    lp_only = labels[:, 0] & ~labels[:, 1]
    vp_only = ~labels[:, 0] & labels[:, 1]
    lp_count, vp_count = int(lp_only.sum()), int(vp_only.sum())
    if min(lp_count, vp_count) == 0:
        raise ValueError("validation needs both LP-only and VP-only failure examples")
    current_failure = torch.where(masks == 1, labels[:, 0], labels[:, 1])
    safe_count = int((~current_failure).sum())
    candidates = []
    best_key, best_rule = None, None
    for danger in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for safe in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            if safe >= danger:
                continue
            rule = SelectionRule(danger, safe)
            desired, _ = propose_selection(probabilities, masks, rule)
            lp_wrong = int((lp_only & (desired == 1)).sum())
            vp_wrong = int((vp_only & (desired == 0)).sum())
            unnecessary = int((~current_failure & (desired != masks)).sum())
            total_wrong = lp_wrong + vp_wrong
            worst_direction = max(lp_wrong / lp_count, vp_wrong / vp_count)
            # Last two fields provide a deterministic conservative tie-break.
            key = (total_wrong, worst_direction, unnecessary, -danger, safe)
            candidates.append({**rule.to_dict(), "dangerous_branch_count": total_wrong,
                "dangerous_branch_rate": total_wrong / (lp_count + vp_count),
                "lp_only_failure_wrong_rate": lp_wrong / lp_count,
                "vp_only_failure_wrong_rate": vp_wrong / vp_count,
                "unnecessary_switch_count": unnecessary,
                "unnecessary_switch_rate": unnecessary / safe_count if safe_count else None})
            if best_key is None or key < best_key:
                best_key, best_rule = key, rule
    assert best_rule is not None
    return best_rule, {"fit_split": "validation", "uses_test_labels": False,
        "scope": "pointwise_requests_before_confirmation_dwell_and_blend",
        "objective": ["dangerous_branch_count", "worst_directional_wrong_rate",
                      "unnecessary_switch_count"],
        "lp_only_failure_samples": lp_count, "vp_only_failure_samples": vp_count,
        "selected": best_rule.to_dict(), "candidates": candidates}
