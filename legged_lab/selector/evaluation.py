"""Shared prediction/report code for evaluation and validation-only calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .metrics import calibration_diagnostics, evaluation_checks, selector_metrics
from .selection import SelectionRule, current_mask_from_features, propose_selection


@dataclass
class Predictions:
    probabilities: torch.Tensor
    labels: torch.Tensor
    current_mask: torch.Tensor
    audit: dict[str, torch.Tensor]
    outcome_probabilities: torch.Tensor | None = None


@torch.no_grad()
def collect_predictions(model, loader, device) -> Predictions:
    model.eval()
    probabilities, labels, masks, outcomes = [], [], [], []
    audit: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        raw_features = batch["features"]
        masks.append(current_mask_from_features(raw_features).cpu())
        if hasattr(model, "decision_outputs"):
            risk, q = model.decision_outputs(raw_features.to(device, non_blocking=True))
            probabilities.append(risk.float().cpu())
            outcomes.append(q.float().cpu())
        else:
            probabilities.append(model(raw_features.to(device, non_blocking=True)).float().cpu())
        labels.append(batch["labels"].float().cpu())
        for key in ("fault_metadata", "terrain_level", "terrain_type", "episode_id", "future_metrics"):
            if key in batch:
                audit.setdefault(key, []).append(batch[key].cpu())
    if not probabilities:
        raise ValueError("cannot evaluate an empty selector split")
    return Predictions(torch.cat(probabilities), torch.cat(labels), torch.cat(masks),
                       {key: torch.cat(values) for key, values in audit.items()},
                       torch.cat(outcomes) if outcomes else None)


def report_predictions(
    predictions: Predictions,
    rule: SelectionRule | None,
    *,
    min_class_samples: int = 20,
    min_contrast_samples: int = 50,
    max_error_cases: int = 100,
    outcome_guard_config=None,
) -> dict[str, Any]:
    p, y, current = predictions.probabilities, predictions.labels, predictions.current_mask
    q = predictions.outcome_probabilities
    kwargs = {"current_mask": current, "selection_rule": rule, "outcome_probabilities": q}
    metrics = selector_metrics(p, y, **kwargs)
    breakdown = {}

    def add(name, selected):
        if selected.any():
            breakdown[name] = selector_metrics(p[selected], y[selected],
                current_mask=current[selected], selection_rule=rule,
                outcome_probabilities=q[selected] if q is not None else None)

    if "fault_metadata" in predictions.audit:
        faults = predictions.audit["fault_metadata"].long()
        lp_active, vp_active = faults[:, 6].bool(), faults[:, 7].bool()
        add("normal", ~lp_active & ~vp_active)
        add("lp_fault", lp_active & ~vp_active)
        add("vp_fault", ~lp_active & vp_active)
        add("dual_fault", lp_active & vp_active)
        add("fault_recovery", (faults[:, 3] > 0) & ((faults[:, 0] > 0) | (faults[:, 1] > 0)))
        for head, name, active in ((0, "lp", lp_active), (1, "vp", vp_active)):
            for kind in torch.unique(faults[active, head]).tolist():
                add(f"active_{name}_fault_kind_{int(kind)}", active & (faults[:, head] == kind))
    for key in ("terrain_level", "terrain_type"):
        if key in predictions.audit:
            values = predictions.audit[key].long()
            for value in torch.unique(values).tolist():
                add(f"{key}_{int(value)}", values == value)
    if "future_metrics" in predictions.audit:
        first = predictions.audit["future_metrics"][:, :, 0].long()
        for head, name in enumerate(("lp", "vp")):
            positive = y[:, head].bool()
            add(f"{name}_failure_within_100ms", positive & (first[:, head] >= 1) & (first[:, head] <= 5))
            add(f"{name}_failure_after_100ms", positive & (first[:, head] > 5))

    if rule is None:
        selected = (p[:, 0] + .05 < p[:, 1]).float()
    else:
        selected, _ = propose_selection(p, current, rule, outcome_probabilities=q)
    lp_only = y[:, 0].bool() & ~y[:, 1].bool()
    vp_only = ~y[:, 0].bool() & y[:, 1].bool()
    wrong = (lp_only & (selected == 1)) | (vp_only & (selected == 0))
    cases = {}
    # Cap each direction separately so the rarer VP failures remain visible.
    for name, direction in (("lp_only_failure", lp_only), ("vp_only_failure", vp_only)):
        indices = (wrong & direction).nonzero().flatten()[:max_error_cases].tolist()
        examples = []
        for index in indices:
            row = {"split_row_index": index, "labels": y[index].int().tolist(),
                "risks": p[index].tolist(), "current_mask": int(current[index]),
                "selected_mask": int(selected[index])}
            if q is not None:
                row["outcome_probabilities"] = q[index].tolist()
            for key in ("fault_metadata", "terrain_level", "terrain_type", "episode_id"):
                if key in predictions.audit:
                    row[key] = predictions.audit[key][index].tolist()
            if "future_metrics" in predictions.audit:
                row["first_failure_step_lp_vp"] = predictions.audit["future_metrics"][index, :, 0].tolist()
            examples.append(row)
        cases[name] = examples
    checks = evaluation_checks(metrics, min_class_samples=min_class_samples,
                               min_contrast_samples=min_contrast_samples)
    if rule is not None and rule.mode == "learned_outcomes":
        from dataclasses import replace
        from .outcome_decision import OutcomeGuardConfig, outcome_evaluation_checks
        cfg = replace(outcome_guard_config or OutcomeGuardConfig(),
                      min_class_samples=min_class_samples, min_contrast_samples=min_contrast_samples)
        checks.update(outcome_evaluation_checks(metrics, cfg))
    outcome_diagnostics = None
    if q is not None:
        from .outcomes import OUTCOME_ORDER, outcome_classes
        target = outcome_classes(y)
        confusion = torch.bincount(target*4+q.argmax(1), minlength=16).reshape(4,4)
        outcome_diagnostics = {"class_order": list(OUTCOME_ORDER), "confusion_rows_true_columns_predicted": confusion.tolist(),
            "joint_warning_is_switch_veto": False}
    return {"samples": len(y), "metrics": metrics, "breakdown": breakdown,
        "checks": checks,
        "rank_comparison": selector_metrics(p, y),
        "calibration_bins": calibration_diagnostics(p, y),
        "directional_error_examples": cases,
        "outcome_diagnostics": outcome_diagnostics,
        "selection_rule": rule.to_dict() if rule else None,
        "selection_scope": (f"pointwise_{rule.mode}_requests_using_observed_current_branch"
                            if rule else "legacy_rank_rule_lp_risk_plus_0.05_less_than_vp_risk"),
        "runtime_gates_evaluated": False,
        "notes": ["Counts are rows and may contain correlated samples from one event.",
                  "These are offline one-second label outcomes, not online rollout fall rates.",
                  "Confirmation, dwell, health debounce and subsequent closed-loop states require rollout validation.",
                  "fault_recovery is a subset and must not be added to the four disjoint fault scenarios."]}
