"""Dependency-free probability and branch-selection metrics."""

from __future__ import annotations

import math

import torch

from .selection import SelectionRule, propose_selection


@torch.no_grad()
def binary_auroc(probability: torch.Tensor, label: torch.Tensor) -> float:
    probability = probability.reshape(-1).float().cpu()
    label = label.reshape(-1).bool().cpu()
    positive = int(label.sum())
    negative = int((~label).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(probability, stable=True)
    sorted_probability = probability[order]
    # Equal scores must receive their average rank; arbitrary ordering of ties
    # otherwise makes AUROC depend on input order.
    _, counts = torch.unique_consecutive(sorted_probability, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts + 1
    ranks_sorted = torch.repeat_interleave((starts + ends).double() / 2, counts)
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    positive_rank_sum = ranks[label].sum()
    auc = (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    return float(auc)


@torch.no_grad()
def expected_calibration_error(
    probability: torch.Tensor, label: torch.Tensor, bins: int = 15
) -> float:
    probability = probability.reshape(-1).float().cpu().clamp(0.0, 1.0)
    label = label.reshape(-1).float().cpu()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    error = torch.tensor(0.0)
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            error += selected.float().mean() * (
                probability[selected].mean() - label[selected].mean()
            ).abs()
    return float(error)


@torch.no_grad()
def selector_metrics(
    probabilities: torch.Tensor, labels: torch.Tensor, *,
    current_mask: torch.Tensor | None = None,
    selection_rule: SelectionRule | None = None,
    outcome_probabilities: torch.Tensor | None = None,
) -> dict[str, float]:
    if (probabilities.ndim != 2 or probabilities.shape != labels.shape
            or probabilities.shape[1] != 2 or probabilities.shape[0] == 0):
        raise ValueError("probabilities and labels must have matching [samples, 2] shapes")
    if (not torch.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("risk probabilities must be finite and in [0, 1]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("risk labels must be binary")
    names = ("lp_1s", "vp_1s")
    metrics: dict[str, float] = {}
    metrics["coverage/samples"] = int(labels.shape[0])
    for index, name in enumerate(names):
        positive = int(labels[:, index].sum())
        metrics[f"coverage/{name}_positive"] = positive
        metrics[f"coverage/{name}_negative"] = int(labels.shape[0]) - positive
        metrics[f"auroc/{name}"] = binary_auroc(probabilities[:, index], labels[:, index])
        metrics[f"ece/{name}"] = expected_calibration_error(
            probabilities[:, index], labels[:, index]
        )
        metrics[f"calibration/{name}_mean_risk"] = float(probabilities[:, index].mean())
        metrics[f"calibration/{name}_observed_failure_rate"] = positive / len(labels)
        metrics[f"calibration/{name}_signed_mean_error"] = (
            metrics[f"calibration/{name}_mean_risk"] - positive / len(labels)
        )
        metrics[f"brier/{name}"] = float((probabilities[:, index] - labels[:, index]).square().mean())
    lp_risk, vp_risk = probabilities[:, 0], probabilities[:, 1]
    if selection_rule is None:
        # Retained unchanged for comparison with the first training report.
        choose_lp = lp_risk + 0.05 < vp_risk
        both_flag = None
    else:
        if current_mask is None:
            raise ValueError("danger-to-safe evaluation needs the observed current branch")
        desired, both_flag = propose_selection(probabilities, current_mask, selection_rule,
                                                outcome_probabilities=outcome_probabilities)
        choose_lp = desired == 1
    lp_failure = labels[:, 0].bool()
    vp_failure = labels[:, 1].bool()
    selected_failure = torch.where(choose_lp, lp_failure, vp_failure)
    oracle_failure = lp_failure & vp_failure
    one_safe = lp_failure ^ vp_failure
    dangerous_choice = one_safe & torch.where(choose_lp, lp_failure, vp_failure)
    metrics["coverage/lp_only_failure"] = int((lp_failure & ~vp_failure).sum())
    metrics["coverage/vp_only_failure"] = int((~lp_failure & vp_failure).sum())
    metrics["coverage/both_safe"] = int((~lp_failure & ~vp_failure).sum())
    metrics["coverage/both_failure"] = int(oracle_failure.sum())
    contrasting_samples = int(one_safe.sum())
    metrics["coverage/contrasting_samples"] = contrasting_samples
    metrics["failure/always_lp"] = float(lp_failure.float().mean())
    metrics["failure/always_vp"] = float(vp_failure.float().mean())
    metrics["failure/selector"] = float(selected_failure.float().mean())
    metrics["failure/oracle"] = float(oracle_failure.float().mean())
    best_fixed = min(metrics["failure/always_lp"], metrics["failure/always_vp"])
    metrics["failure/improvement_over_best_fixed"] = (
        (best_fixed - metrics["failure/selector"]) / best_fixed
        if best_fixed > 0.0
        else 0.0
    )
    metrics["selection/dangerous_branch_rate"] = (
        float(dangerous_choice.sum()) / contrasting_samples
        if contrasting_samples else float("nan")
    )
    metrics["selection/dangerous_branch_count"] = int(dangerous_choice.sum())
    for name, contrast, wrong in (
        ("lp", lp_failure & ~vp_failure, choose_lp),
        ("vp", ~lp_failure & vp_failure, ~choose_lp),
    ):
        count = int(contrast.sum())
        wrong_count = int((contrast & wrong).sum())
        metrics[f"selection/{name}_only_failure_wrong_count"] = wrong_count
        metrics[f"selection/{name}_only_failure_wrong_rate"] = wrong_count / count if count else float("nan")
        metrics[f"selection/{name}_only_failure_correct_count"] = count - wrong_count
    if current_mask is not None:
        mask = current_mask.to(probabilities.device).reshape(-1)
        if mask.shape != (len(labels),) or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("current_mask must contain one binary mask per sample")
        current_lp = mask == 1
        changed = choose_lp != current_lp
        current_failure = torch.where(current_lp, lp_failure, vp_failure)
        alternative_failure = torch.where(current_lp, vp_failure, lp_failure)
        rescue = current_failure & ~alternative_failure
        safe_current = ~current_failure
        harmful_opportunity = safe_current & alternative_failure
        for name, selected, denominator in (
            ("missed_rescue", rescue & ~changed, rescue),
            ("correct_rescue", rescue & changed, rescue),
            ("unnecessary_switch", safe_current & changed, safe_current),
            ("harmful_switch", harmful_opportunity & changed, harmful_opportunity),
            ("both_safe_switch", ~lp_failure & ~vp_failure & changed, ~lp_failure & ~vp_failure),
            ("both_failure_switch", oracle_failure & changed, oracle_failure),
        ):
            count, total = int(selected.sum()), int(denominator.sum())
            metrics[f"switch/{name}_count"] = count
            metrics[f"switch/{name}_samples"] = total
            metrics[f"switch/{name}_rate"] = count / total if total else float("nan")
        metrics["switch/request_count"] = int(changed.sum())
        metrics["switch/request_fraction"] = float(changed.float().mean())
    if both_flag is not None:
        flagged = int(both_flag.sum())
        caught = int((both_flag & oracle_failure).sum())
        metrics["selection/both_unreliable_flag_count"] = flagged
        metrics["selection/both_failure_flag_recall"] = caught / int(oracle_failure.sum()) if oracle_failure.any() else float("nan")
        metrics["selection/both_unreliable_flag_precision"] = caught / flagged if flagged else float("nan")
        metrics["selection/both_failure_flag_true_positive_count"] = caught
        metrics["selection/both_failure_flag_false_positive_count"] = flagged-caught
        metrics["selection/both_failure_flag_false_negative_count"] = int(oracle_failure.sum())-caught
        metrics["selection/both_failure_flag_single_failure_count"] = int((both_flag & one_safe).sum())
        metrics["selection/both_failure_flag_both_safe_count"] = int((both_flag & ~lp_failure & ~vp_failure).sum())
    if selection_rule is not None:
        if selection_rule.mode == "learned_outcomes":
            from .outcome_decision import outcome_masks
            low_hold, high_hold, _ = outcome_masks(outcome_probabilities, selection_rule)
        elif selection_rule.mode == "directional_comparison":
            low_hold = (probabilities <= selection_rule.both_safe_hold_risk).all(1)
            high_hold = (probabilities >= selection_rule.both_unreliable_risk).all(1)
        else:
            low_hold = (probabilities <= selection_rule.safe_threshold).all(1)
            high_hold = (probabilities >= selection_rule.danger_threshold).all(1)
        metrics["selection/low_risk_hold_count"] = int(low_hold.sum())
        metrics["selection/low_risk_hold_true_safe_count"] = int((low_hold & ~lp_failure & ~vp_failure).sum())
        metrics["selection/both_failure_hold_count"] = int(high_hold.sum())
        metrics["selection/both_failure_hold_true_positive_count"] = int((high_hold & oracle_failure).sum())
        metrics["selection/both_failure_hold_single_failure_count"] = int((high_hold & one_safe).sum())
    if outcome_probabilities is not None:
        from .outcomes import OUTCOME_ORDER, outcome_classes, validate_outcome_probabilities
        validate_outcome_probabilities(outcome_probabilities, len(labels))
        q = outcome_probabilities.to(probabilities.device).float()
        target = outcome_classes(labels)
        metrics["outcome/log_loss"] = float(-q.clamp_min(1e-12).log().gather(1, target[:,None]).mean())
        metrics["outcome/accuracy"] = float((q.argmax(1) == target).float().mean())
        metrics["outcome/brier"] = float((q-torch.nn.functional.one_hot(target,4)).square().sum(1).mean())
        for column, name in enumerate(OUTCOME_ORDER):
            metrics[f"outcome/auroc/{name}"] = binary_auroc(q[:,column], target == column)
            metrics[f"outcome/ece/{name}"] = expected_calibration_error(q[:,column], target == column)
        if one_safe.any():
            pair = q[one_safe,1]/q[one_safe,1:3].sum(1).clamp_min(1e-12)
            metrics["outcome/conditional_lp_failure_auroc"] = binary_auroc(pair, target[one_safe] == 1)
    metrics["selection/lp_fraction"] = float(choose_lp.float().mean())
    finite_aurocs = [value for key, value in metrics.items() if key.startswith("auroc/") and math.isfinite(value)]
    finite_eces = [value for key, value in metrics.items() if key.startswith("ece/") and math.isfinite(value)]
    metrics["auroc/min"] = min(finite_aurocs) if len(finite_aurocs) == 2 else float("nan")
    metrics["ece/max"] = max(finite_eces) if len(finite_eces) == 2 else float("nan")
    return metrics


@torch.no_grad()
def calibration_diagnostics(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> dict:
    """Reliability bins include counts; ECE is not a classification error rate."""
    result = {}
    for head, name in enumerate(("lp_1s", "vp_1s")):
        p, y = probabilities[:, head].cpu(), labels[:, head].float().cpu()
        rows = []
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            mask = (p >= lower) & ((p <= upper) if index == bins-1 else (p < upper))
            count = int(mask.sum())
            rows.append({"lower": lower, "upper": upper, "samples": count,
                         "mean_risk": float(p[mask].mean()) if count else None,
                         "observed_failure_rate": float(y[mask].mean()) if count else None})
        result[name] = rows
    return result


def evaluation_checks(
    metrics: dict[str, float], *, min_class_samples: int = 20, min_contrast_samples: int = 50
) -> dict[str, bool]:
    """Require measurable heads and both branch contrasts before passing gates.

    Counts are minimum coverage checks, not statistical confidence guarantees.
    """
    if min_class_samples < 1 or min_contrast_samples < 1:
        raise ValueError("evaluation coverage thresholds must be positive")
    names = ("lp_1s", "vp_1s")
    heads_covered = all(
        metrics[f"coverage/{name}_{kind}"] >= min_class_samples
        for name in names for kind in ("positive", "negative")
    )
    contrasts_covered = all(
        metrics[f"coverage/{name}_only_failure"] >= min_contrast_samples
        for name in ("lp", "vp")
    )
    return {
        "all_heads_have_class_coverage": heads_covered,
        "both_branch_contrasts_have_coverage": contrasts_covered,
        "all_auroc_at_least_0.85": heads_covered and metrics["auroc/min"] >= 0.85,
        "all_ece_at_most_0.05": heads_covered and metrics["ece/max"] <= 0.05,
        "dangerous_branch_rate_at_most_0.03": (
            contrasts_covered and metrics["selection/dangerous_branch_rate"] <= 0.03
        ),
    }


def json_safe_metrics(value):
    """Write undefined metrics as JSON null, never non-standard NaN literals."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_metrics(item) for item in value]
    return value
