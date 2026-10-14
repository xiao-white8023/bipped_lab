"""Validation-only joint-warning fitting and learned branch-selection rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
import torch

from .outcomes import outcome_classes, validate_outcome_probabilities
from .selection import SelectionRule


@dataclass(frozen=True)
class OutcomeGuardConfig:
    max_dangerous_rate: float = 0.03
    max_both_safe_switch_rate: float = 0.05
    max_both_failure_switch_rate: float = 0.10
    min_both_failure_recall: float = 0.80
    min_both_failure_precision: float = 0.50
    min_class_samples: int = 20
    min_contrast_samples: int = 50

    def validate(self):
        for name in ("max_dangerous_rate", "max_both_safe_switch_rate", "max_both_failure_switch_rate",
                     "min_both_failure_recall", "min_both_failure_precision"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.min_class_samples < 1 or self.min_contrast_samples < 1:
            raise ValueError("outcome coverage requirements must be positive")


def outcome_evaluation_checks(metrics: dict, cfg: OutcomeGuardConfig | None = None):
    cfg = cfg or OutcomeGuardConfig()
    cfg.validate()
    covered = all(metrics[f"coverage/{k}"] >= cfg.min_class_samples
                  for k in ("both_safe", "both_failure", "lp_only_failure", "vp_only_failure"))
    contrasting = all(metrics[f"coverage/{k}_only_failure"] >= cfg.min_contrast_samples for k in ("lp", "vp"))
    return {
        "all_four_outcomes_have_coverage": covered,
        "both_branch_contrasts_have_coverage": contrasting,
        "dangerous_branch_rate_within_target": contrasting and metrics["selection/dangerous_branch_rate"] <= cfg.max_dangerous_rate,
        "both_safe_switch_rate_within_target": covered and metrics["switch/both_safe_switch_rate"] <= cfg.max_both_safe_switch_rate,
        "both_failure_switch_rate_within_target": covered and metrics["switch/both_failure_switch_rate"] <= cfg.max_both_failure_switch_rate,
        "both_failure_warning_recall_within_target": covered and metrics["selection/both_failure_flag_recall"] >= cfg.min_both_failure_recall,
        "both_failure_warning_precision_within_target": covered and metrics["selection/both_unreliable_flag_precision"] >= cfg.min_both_failure_precision,
        "low_risk_hold_effective": metrics.get("selection/low_risk_hold_true_safe_count", 0) > 0,
        "both_failure_hold_effective": metrics.get("selection/both_failure_hold_true_positive_count", 0) > 0,
        "joint_failure_probability_calibrated": covered and metrics.get("outcome/ece/both_failure", math.inf) <= .05,
    }


def outcome_guard(baseline: dict, metrics: dict, cfg: OutcomeGuardConfig):
    checks = outcome_evaluation_checks(metrics, cfg)
    checks["risk_auroc_at_least_0.85"] = metrics["auroc/min"] >= .85
    checks["risk_ece_at_most_0.05"] = metrics["ece/max"] <= .05
    # Keep the previously agreed directional and rescue protections.
    for name, key in (
        ("dangerous_count_not_worse", "selection/dangerous_branch_count"),
        ("lp_only_errors_not_worse", "selection/lp_only_failure_wrong_count"),
        ("vp_only_errors_not_worse", "selection/vp_only_failure_wrong_count"),
        ("missed_rescues_not_worse", "switch/missed_rescue_count"),
        ("harmful_switches_not_worse", "switch/harmful_switch_count"),
        ("both_failure_switches_not_worse", "switch/both_failure_switch_count"),
    ):
        checks[name] = metrics[key] <= baseline[key]
    return {"passed": all(checks.values()), "checks": checks, "fit_split": "validation",
            "test_used_for_acceptance": False, "requirements": asdict(cfg)}


def validate_outcome_guard_record(guard):
    required = {
        "all_four_outcomes_have_coverage", "both_branch_contrasts_have_coverage", "dangerous_branch_rate_within_target",
        "both_safe_switch_rate_within_target", "both_failure_switch_rate_within_target",
        "both_failure_warning_recall_within_target", "both_failure_warning_precision_within_target",
        "low_risk_hold_effective", "both_failure_hold_effective", "joint_failure_probability_calibrated",
        "risk_auroc_at_least_0.85", "risk_ece_at_most_0.05", "dangerous_count_not_worse",
        "lp_only_errors_not_worse", "vp_only_errors_not_worse", "missed_rescues_not_worse",
        "harmful_switches_not_worse", "both_failure_switches_not_worse",
    }
    if (not isinstance(guard,dict) or guard.get("passed") is not True or guard.get("fit_split") != "validation"
            or guard.get("test_used_for_acceptance") is not False or not isinstance(guard.get("checks"),dict)
            or not required <= guard["checks"].keys() or any(v is not True for v in guard["checks"].values())):
        raise ValueError("learned outcome artifact did not pass its complete validation guard")
    requirements = guard.get("requirements")
    if not isinstance(requirements,dict) or set(requirements) != set(asdict(OutcomeGuardConfig())):
        raise ValueError("outcome validation guard is missing its explicit requirements")
    OutcomeGuardConfig(**requirements).validate()


def outcome_masks(q: torch.Tensor, rule: SelectionRule):
    low_hold = q[:, 0] >= rule.both_safe_probability_threshold
    both_hold = q[:, 3] >= rule.both_failure_hold_threshold
    warning = q[:, 3] >= rule.both_failure_flag_threshold
    return low_hold, both_hold, warning


def propose_outcome_selection(risks, q, mask, rule, lp_ok, vp_ok):
    validate_outcome_probabilities(q, len(risks))
    q = q.to(risks.device)
    low_hold, both_hold, warning = outcome_masks(q, rule)
    gap = q[:, 1].clamp_min(1e-12).log() - q[:, 2].clamp_min(1e-12).log()
    eligible = lp_ok & vp_ok & ~low_hold & ~both_hold
    desired = mask.clone()
    to_vp = ((mask == 1) & eligible & (q[:, 1] >= rule.lp_to_vp_min_rescue_probability)
             & (gap > rule.lp_to_vp_margin))
    to_lp = ((mask == 0) & eligible & (q[:, 2] >= rule.vp_to_lp_min_rescue_probability)
             & (-gap > rule.vp_to_lp_margin))
    # Sensor failures still use ordinary confirmation, dwell and blend gates.
    to_vp |= (mask == 1) & ~lp_ok & vp_ok & ~both_hold & (risks[:, 1] <= .1)
    to_lp |= (mask == 0) & ~vp_ok & lp_ok & ~both_hold & (risks[:, 0] <= .1)
    desired[to_vp] = 0
    desired[to_lp] = 1
    warning = warning | (~lp_ok & ~vp_ok) | (~lp_ok & (risks[:, 1] > .1)) | (~vp_ok & (risks[:, 0] > .1))
    return desired, warning


def fit_joint_warning(probability, both_failed, cfg):
    """Use actual validation precision/recall; zero warnings can never pass."""
    p = probability.detach().cpu().float().double().numpy()
    target = both_failed.detach().cpu().bool().numpy()
    positives = int(target.sum())
    if not positives or positives == len(target):
        raise ValueError("joint warning fitting needs both positive and negative validation examples")
    order = np.argsort(-p, kind="stable")
    values, y = p[order], target[order]
    ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    caught = np.cumsum(y, dtype=np.int64)[ends]
    count = ends + 1
    recall, precision = caught/positives, caught/count
    feasible = (recall >= cfg.min_both_failure_recall) & (precision >= cfg.min_both_failure_precision)
    if feasible.any():
        # Smallest alert set that reaches the required recall and precision.
        chosen = int(np.flatnonzero(feasible)[0])
    else:
        f1 = 2*recall*precision/np.maximum(recall+precision, 1e-12)
        chosen = int(np.argmax(f1))
    boundary = int(ends[chosen])
    threshold = float(values[boundary])
    # Prefer a boundary between distinct FP32 scores, rather than resting on
    # one validation row. This keeps the same alert set and reduces sensitivity
    # to small device/export arithmetic differences.
    if boundary+1 < len(values):
        lower = float(values[boundary+1])
        midpoint = float(np.float32((threshold+lower)/2))
        if lower < midpoint <= threshold:
            threshold = midpoint
    return threshold, {
        "fit_split": "validation", "uses_test_labels": False,
        "passed": bool(feasible[chosen]), "threshold": threshold,
        "flagged": int(count[chosen]), "true_positives": int(caught[chosen]),
        "recall": float(recall[chosen]), "precision": float(precision[chosen]),
        "required_recall": cfg.min_both_failure_recall, "required_precision": cfg.min_both_failure_precision,
        "warning_is_switch_veto": False,
    }


def _options(q, y, masks, gap, current_lp, safe_threshold, failed_threshold, margins):
    selected = masks == int(current_lp)
    s = gap[selected] if current_lp else -gap[selected]
    order = np.argsort(s, kind="stable")
    s, here = s[order], q[selected][order]
    branch = 0 if current_lp else 1
    failure, other = y[selected, branch][order], y[selected, 1-branch][order]
    rescue, harmful = failure & ~other, ~failure & other
    both_safe, both_failed = ~failure & ~other, failure & other
    rescue_probability = here[:, 1 if current_lp else 2]
    shared = (here[:, 0] < safe_threshold) & (here[:, 3] < failed_threshold)
    cuts = np.searchsorted(s, np.asarray(margins, dtype=np.float32), side="right")
    base = int(rescue.sum())
    output = []

    def suffix(values):
        c = np.r_[0, np.cumsum(values, dtype=np.int64)]
        return c[-1]-c[cuts]

    for minimum in (0., .001, .003, .01, .03, .10, .20):
        eligible = shared & (rescue_probability >= minimum)
        saved = suffix(eligible & rescue)
        bad = suffix(eligible & harmful)
        safe = suffix(eligible & both_safe)
        dual = suffix(eligible & both_failed)
        for i, margin in enumerate(margins):
            output.append({"minimum": minimum, "margin": float(margin), "missed": base-int(saved[i]),
                           "harmful": int(bad[i]), "both_safe": int(safe[i]), "both_failed": int(dual[i])})
    return output


def _best_pair(lp, vp, baseline, counts, cfg, common_passed):
    def array(rows, name):
        return np.asarray([x[name] for x in rows])
    lm, vm = array(lp, "missed")[:, None], array(vp, "missed")[None, :]
    lb, vb = array(lp, "harmful")[:, None], array(vp, "harmful")[None, :]
    lw, vw = lm+vb, vm+lb
    wrong, missed, harmful = lw+vw, lm+vm, lb+vb
    safe = array(lp, "both_safe")[:, None] + array(vp, "both_safe")[None, :]
    dual = array(lp, "both_failed")[:, None] + array(vp, "both_failed")[None, :]
    redundant = safe + harmful + dual
    compatible = array(lp, "margin")[:, None] + array(vp, "margin")[None, :] >= 0
    accepted = compatible & common_passed
    accepted &= wrong <= math.floor(cfg.max_dangerous_rate*(counts[1]+counts[2]))
    accepted &= wrong <= baseline["selection/dangerous_branch_count"]
    accepted &= lw <= baseline["selection/lp_only_failure_wrong_count"]
    accepted &= vw <= baseline["selection/vp_only_failure_wrong_count"]
    accepted &= missed <= baseline["switch/missed_rescue_count"]
    accepted &= harmful <= baseline["switch/harmful_switch_count"]
    accepted &= dual <= min(baseline["switch/both_failure_switch_count"], math.floor(cfg.max_both_failure_switch_rate*counts[3]))
    accepted &= safe <= math.floor(cfg.max_both_safe_switch_rate*counts[0])

    def best(where):
        indices = np.flatnonzero(where)
        if not len(indices):
            return None
        worst_direction = np.maximum(lw/counts[1], vw/counts[2])
        order = np.lexsort((indices, redundant.ravel()[indices], missed.ravel()[indices],
                            worst_direction.ravel()[indices], wrong.ravel()[indices]))
        index = int(indices[order[0]])
        i, j = np.unravel_index(index, wrong.shape)
        return {"lp": lp[i], "vp": vp[j], "wrong": int(wrong[i,j]), "missed": int(missed[i,j]),
                "harmful": int(harmful[i,j]), "both_safe": int(safe[i,j]), "both_failed": int(dual[i,j]),
                "redundant": int(redundant[i,j]), "worst_direction_rate": float(worst_direction[i,j])}
    return best(accepted), best(compatible), int(compatible.sum())


@torch.no_grad()
def fit_outcome_rule(risks, q, labels, masks, baseline_probabilities, cfg=None):
    from .metrics import selector_metrics
    cfg = cfg or OutcomeGuardConfig()
    cfg.validate()
    risks, q, labels, masks = [v.detach().cpu().float() for v in (risks, q, labels, masks)]
    validate_outcome_probabilities(q, len(labels))
    baseline = selector_metrics(baseline_probabilities.cpu(), labels, current_mask=masks)
    risk_quality = selector_metrics(risks, labels, current_mask=masks)
    counts = torch.bincount(outcome_classes(labels), minlength=4).tolist()
    if min(counts) == 0:
        raise ValueError("decision validation needs all four outcomes")
    warning_threshold, warning_fit = fit_joint_warning(q[:, 3], outcome_classes(labels) == 3, cfg)
    gap = (q[:,1].clamp_min(1e-12).log()-q[:,2].clamp_min(1e-12).log()).numpy().astype(np.float32)
    # Validation-derived score quantiles refine the grid; test scores are absent.
    contrast = outcome_classes(labels).numpy()
    contrast_scores = gap[(contrast == 1) | (contrast == 2)]
    fixed = np.asarray([-4.,-3.,-2.,-1.5,-1.,-.75,-.5,-.25,0.,.1,.2,.3,.5,.75,1.,1.5,2.,3.,4.], dtype=np.float32)
    empirical = np.quantile(contrast_scores, np.linspace(.02,.98,31)).clip(-8.,8.).astype(np.float32)
    margins = np.unique(np.r_[fixed, empirical, -empirical]).astype(np.float32)
    accepted_best, accuracy_best, summaries, candidates = None, None, [], 0
    q_np, y_np, mask_np = q.numpy(), labels.numpy().astype(bool), masks.reshape(-1).numpy()
    coverage = min(counts) >= cfg.min_class_samples and min(counts[1:3]) >= cfg.min_contrast_samples
    joint_ece = None
    from .metrics import expected_calibration_error
    joint_ece = expected_calibration_error(q[:,3], outcome_classes(labels) == 3)
    hold_grid = sorted({.2,.4,.6,.8,.95,1., warning_threshold})
    for safe_threshold in (.90,.95,.98,.99,.995,1.):
        for failed_threshold in hold_grid:
            if failed_threshold < warning_threshold:
                continue
            low_effective = bool(((q[:,0] >= safe_threshold) & (outcome_classes(labels) == 0)).any())
            high_effective = bool(((q[:,3] >= failed_threshold) & (outcome_classes(labels) == 3)).any())
            common = (coverage and warning_fit["passed"] and low_effective and high_effective and joint_ece <= .05
                      and risk_quality["auroc/min"] >= .85 and risk_quality["ece/max"] <= .05)
            lp = _options(q_np,y_np,mask_np,gap,True,safe_threshold,failed_threshold,margins)
            vp = _options(q_np,y_np,mask_np,gap,False,safe_threshold,failed_threshold,margins)
            accepted, accuracy, count = _best_pair(lp,vp,baseline,counts,cfg,common)
            candidates += count
            for row in (accepted,accuracy):
                if row is not None:
                    row.update({"safe_threshold": safe_threshold, "failed_threshold": failed_threshold})
            for row, kind in ((accepted,"accepted"),(accuracy,"accuracy")):
                if row is None:
                    continue
                key = (row["wrong"], row["worst_direction_rate"], row["missed"], row["redundant"])
                previous = accepted_best if kind == "accepted" else accuracy_best
                if previous is None or key < previous[0]:
                    if kind == "accepted": accepted_best = (key,row)
                    else: accuracy_best = (key,row)
            summaries.append({"safe_threshold": safe_threshold, "failed_threshold": failed_threshold,
                              "best_accepted": accepted, "best_accuracy": accuracy})
    chosen = (accepted_best or accuracy_best)[1]
    rule = SelectionRule(mode="learned_outcomes", both_safe_probability_threshold=chosen["safe_threshold"],
        both_failure_flag_threshold=warning_threshold, both_failure_hold_threshold=chosen["failed_threshold"],
        lp_to_vp_min_rescue_probability=chosen["lp"]["minimum"], vp_to_lp_min_rescue_probability=chosen["vp"]["minimum"],
        lp_to_vp_margin=chosen["lp"]["margin"], vp_to_lp_margin=chosen["vp"]["margin"])
    measured = selector_metrics(risks, labels, current_mask=masks, selection_rule=rule, outcome_probabilities=q)
    guard = outcome_guard(baseline, measured, cfg)
    if measured["selection/dangerous_branch_count"] != chosen["wrong"]:
        raise RuntimeError("outcome threshold search disagrees with production decisions")
    return rule, {"fit_split": "validation", "uses_test_labels": False, "accepted": guard["passed"],
        "guard": guard, "selected": rule.to_dict(), "baseline_metrics": baseline, "selected_metrics": measured,
        "joint_warning_fit": warning_fit, "margin_grid": margins.tolist(), "compatible_candidates_evaluated": candidates,
        "search_summary": summaries, "scope": "pointwise_requests_before_confirmation_dwell_and_blend",
        "objective": ["pass_directional_and_rescue_guards", "pass_effective_joint_warning_and_hold_guards",
                      "minimize_dangerous_choices", "minimize_worst_direction_error_rate", "minimize_missed_rescues", "minimize_redundant_requests"],
        "notes": ["Joint warnings and high-confidence holds are separate thresholds.",
                  "Rejected candidates remain diagnostic and cannot be exported.",
                  "Test results never change the fitted rule or validation-only acceptance."]}
