"""Validation-only directional search with an explicit baseline guard.

Sorted cumulative counts make each threshold search linear in rows, rather
than running the network or computing AUROC for every candidate.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from .metrics import selector_metrics
from .selection import SelectionRule, preference_scores


def decision_guard(baseline: dict, candidate: dict, rule: SelectionRule | None = None) -> dict:
    checks = {
        "dangerous_count_not_worse": candidate["selection/dangerous_branch_count"] <= baseline["selection/dangerous_branch_count"],
        "lp_only_errors_not_worse": candidate["selection/lp_only_failure_wrong_count"] <= baseline["selection/lp_only_failure_wrong_count"],
        "vp_only_errors_not_worse": candidate["selection/vp_only_failure_wrong_count"] <= baseline["selection/vp_only_failure_wrong_count"],
        "missed_rescues_not_worse": candidate["switch/missed_rescue_count"] <= baseline["switch/missed_rescue_count"],
        "harmful_switches_not_worse": candidate["switch/harmful_switch_count"] <= baseline["switch/harmful_switch_count"],
        "both_failure_switches_not_worse": candidate["switch/both_failure_switch_count"] <= baseline["switch/both_failure_switch_count"],
        "dangerous_rate_at_most_0.03": candidate["selection/dangerous_branch_rate"] <= .03,
    }
    if rule is not None:
        checks["low_risk_hold_enabled"] = rule.both_safe_hold_risk > 0
        checks["both_high_hold_enabled"] = rule.both_unreliable_risk < 1
        checks["low_risk_hold_effective"] = candidate.get("selection/low_risk_hold_true_safe_count", 0) > 0
        checks["both_high_hold_effective"] = candidate.get("selection/both_failure_hold_true_positive_count", 0) > 0
    return {"passed": all(checks.values()), "fit_split": "validation", "checks": checks,
            "test_used_for_acceptance": False}


def _direction_options(p, y, masks, scores, current_is_lp, hold, alarm, margins):
    selected = masks == int(current_is_lp)
    current_head = 0 if current_is_lp else 1
    other_head = 1-current_head
    cur, alt = p[selected, current_head], p[selected, other_head]
    gap = (scores[selected, current_head] - scores[selected, other_head]).astype(np.float32)
    order = np.argsort(gap, kind="stable")
    cur, alt, gap = cur[order], alt[order], gap[order]
    failed = y[selected, current_head][order]
    other_failed = y[selected, other_head][order]
    rescue = failed & ~other_failed
    harmful = ~failed & other_failed
    safe = ~failed
    base_rescues = int(rescue.sum())
    shared = ~((cur <= hold) & (alt <= hold)) & ~((cur >= alarm) & (alt >= alarm))
    cuts = np.searchsorted(gap, np.asarray(margins, dtype=np.float32), side="right")
    options = []

    def suffix_counts(mask):
        prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        return prefix[-1]-prefix[cuts]

    for minimum in (0., .001, .01, .05, .10):
        for maximum in (.15, .30, .50, .90, 1.):
            eligible = shared & (cur >= minimum) & (alt <= maximum)
            saved = suffix_counts(eligible & rescue)
            bad = suffix_counts(eligible & harmful)
            unnecessary = suffix_counts(eligible & safe)
            both_failed = suffix_counts(eligible & failed & other_failed)
            for index, margin in enumerate(margins):
                options.append({"minimum": minimum, "maximum": maximum, "margin": margin,
                    "missed": base_rescues-int(saved[index]), "harmful": int(bad[index]),
                    "unnecessary": int(unnecessary[index]), "both_failed_switch": int(both_failed[index])})
    return options


def _pair_options(lp, vp, baseline, contrasts, allow_accept=True):
    def array(options, key):
        return np.asarray([row[key] for row in options])
    l_miss, v_miss = array(lp, "missed")[:, None], array(vp, "missed")[None, :]
    l_bad, v_bad = array(lp, "harmful")[:, None], array(vp, "harmful")[None, :]
    lp_wrong = l_miss + v_bad
    vp_wrong = v_miss + l_bad
    total = lp_wrong + vp_wrong
    missed, harmful = l_miss + v_miss, l_bad + v_bad
    unnecessary = array(lp, "unnecessary")[:, None] + array(vp, "unnecessary")[None, :]
    both_failed = array(lp, "both_failed_switch")[:, None] + array(vp, "both_failed_switch")[None, :]
    redundant = unnecessary + both_failed
    # Opposite directions cannot nominate opposite targets for the same scores.
    compatible = array(lp, "margin")[:, None] + array(vp, "margin")[None, :] >= 0
    accepted = compatible & (total <= math.floor(.03*contrasts))
    accepted &= total <= baseline["selection/dangerous_branch_count"]
    accepted &= lp_wrong <= baseline["selection/lp_only_failure_wrong_count"]
    accepted &= vp_wrong <= baseline["selection/vp_only_failure_wrong_count"]
    accepted &= missed <= baseline["switch/missed_rescue_count"]
    accepted &= harmful <= baseline["switch/harmful_switch_count"]
    accepted &= both_failed <= baseline["switch/both_failure_switch_count"]
    if not allow_accept:
        accepted[:] = False

    def best(where, accepted_only):
        flat = np.flatnonzero(where)
        if not len(flat):
            return None
        # Reduce unnecessary requests only AFTER the safety guards pass.
        primary = redundant if accepted_only else total
        secondary = total if accepted_only else missed
        order = np.lexsort((flat, harmful.ravel()[flat], secondary.ravel()[flat], primary.ravel()[flat]))
        index = int(flat[order[0]])
        i, j = np.unravel_index(index, total.shape)
        return {"lp": lp[i], "vp": vp[j], "wrong": int(total[i, j]),
                "missed": int(missed[i, j]), "harmful": int(harmful[i, j]),
                "unnecessary": int(unnecessary[i, j]), "both_failed_switch": int(both_failed[i, j]),
                "redundant": int(redundant[i, j]), "guard_passed": accepted_only}
    return best(accepted, True), best(compatible, False), int(compatible.sum())


@torch.no_grad()
def fit_directional_rule(probabilities, labels, current_mask, *, baseline_probabilities=None,
                         reference_scale=(1., 1.), reference_bias=(0., 0.)):
    probabilities = probabilities.detach().cpu().float()
    labels = labels.detach().cpu().float()
    current_mask = current_mask.detach().cpu().reshape(-1).float()
    baseline_p = probabilities if baseline_probabilities is None else baseline_probabilities.detach().cpu().float()
    baseline = selector_metrics(baseline_p, labels, current_mask=current_mask)
    # Calling the metrics function validates probabilities, labels and masks.
    selector_metrics(probabilities, labels, current_mask=current_mask)
    if min(baseline["coverage/lp_only_failure"], baseline["coverage/vp_only_failure"]) == 0:
        raise ValueError("validation needs both LP-only and VP-only failure examples")
    p, y, masks = probabilities.numpy(), labels.numpy().astype(bool), current_mask.numpy()
    margins_by_mode = {
        "probability": (-.10, -.05, -.02, -.01, -.005, 0., .001, .003, .005, .01, .02, .05, .10),
        "log_odds": (-4., -2., -1., -.5, 0., .25, .5, 1., 2., 4.),
        "reference_score": (-.10, -.05, -.02, 0., .01, .02, .05, .075, .10),
    }
    modes = ("probability", "log_odds") if baseline_probabilities is None else tuple(margins_by_mode)
    accepted_best, accuracy_best = None, None
    summaries = []
    candidate_count = 0
    contrasts = baseline["coverage/contrasting_samples"]
    for mode in modes:
        print(f"[selector decision] searching comparison={mode}", flush=True)
        template = SelectionRule(mode="directional_comparison", comparison=mode,
            reference_scale=tuple(reference_scale), reference_bias=tuple(reference_bias))
        scores = preference_scores(probabilities, template).numpy()
        for hold in (0., .0001, .001, .003, .01, .03):
            for alarm in (.70, .90, .99, 1.):
                lp = _direction_options(p, y, masks, scores, True, hold, alarm, margins_by_mode[mode])
                vp = _direction_options(p, y, masks, scores, False, hold, alarm, margins_by_mode[mode])
                low_effective = bool(((probabilities <= hold).all(1) & ~labels.bool().any(1)).any())
                high_effective = bool(((probabilities >= alarm).all(1) & labels.bool().all(1)).any())
                accepted, accuracy, count = _pair_options(lp, vp, baseline, contrasts,
                    hold > 0 and alarm < 1 and low_effective and high_effective)
                candidate_count += count
                for candidate in (accepted, accuracy):
                    if candidate is None:
                        continue
                    candidate.update({"comparison": mode, "hold": hold, "alarm": alarm})
                if accepted is not None:
                    key = (accepted["redundant"], accepted["wrong"], accepted["harmful"])
                    if accepted_best is None or key < accepted_best[0]:
                        accepted_best = key, accepted
                if accuracy is not None:
                    key = (accuracy["wrong"], accuracy["missed"], accuracy["harmful"], accuracy["unnecessary"])
                    if accuracy_best is None or key < accuracy_best[0]:
                        accuracy_best = key, accuracy
                summaries.append({"comparison": mode, "hold_risk": hold, "alarm_risk": alarm,
                                  "best_accepted": accepted, "best_accuracy": accuracy})
    chosen = (accepted_best or accuracy_best)[1]
    rule = SelectionRule(mode="directional_comparison", comparison=chosen["comparison"],
        both_safe_hold_risk=chosen["hold"], both_unreliable_risk=chosen["alarm"],
        lp_to_vp_min_current_risk=chosen["lp"]["minimum"],
        vp_to_lp_min_current_risk=chosen["vp"]["minimum"],
        lp_to_vp_max_alternative_risk=chosen["lp"]["maximum"],
        vp_to_lp_max_alternative_risk=chosen["vp"]["maximum"],
        lp_to_vp_margin=chosen["lp"]["margin"], vp_to_lp_margin=chosen["vp"]["margin"],
        reference_scale=tuple(reference_scale), reference_bias=tuple(reference_bias))
    measured = selector_metrics(probabilities, labels, current_mask=current_mask, selection_rule=rule)
    guard = decision_guard(baseline, measured, rule)
    # Compare the selected rule through the shared production function too.
    if measured["selection/dangerous_branch_count"] != chosen["wrong"]:
        raise RuntimeError("histogram search disagrees with the production selection rule")
    return rule, {"fit_split": "validation", "uses_test_labels": False,
        "scope": "pointwise_requests_before_confirmation_dwell_and_blend",
        "selected": rule.to_dict(), "accepted": guard["passed"], "guard": guard,
        "baseline_metrics": baseline, "selected_metrics": measured,
        "objective": ["pass_baseline_and_3_percent_guards", "minimize_safe_or_both_failed_requests",
                      "minimize_dangerous_choices"],
        "comparison_scores_are_failure_probabilities": rule.comparison == "probability",
        "compatible_candidates_evaluated": candidate_count, "search_summary": summaries,
        "notes": ["Separate direction thresholds must not overlap.",
                  "A rejected candidate is diagnostic only; retain the source checkpoint.",
                  "Acceptance is not a claim about closed-loop robot performance."]}
