"""Reproduce old local V5 diagnostics using ONLY its cached validation rows.

No training, simulation, threshold search, or test-data access. The fixed 0.01
gain diagnostic is illustrative and is not a proposed/deployed selection rule.
Run with the repository's torch-capable Python; JSON is printed to stdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legged_lab.selector.metrics import binary_auroc, expected_calibration_error
from legged_lab.selector.outcome_decision import propose_outcome_selection
from legged_lab.selector.selection import SelectionRule


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(count, total):
    return count / total if total else None


def binary_metrics(score, target):
    return {
        "auroc": binary_auroc(score, target),
        "ece_15_equal_width_bins": expected_calibration_error(score, target),
        "brier": float((score - target.float()).square().mean()),
        "mean_predicted": float(score.mean()),
        "observed_rate": float(target.float().mean()),
    }


def main():
    torch.set_num_threads(4)
    folder = ROOT / "outputs/g1_renet_selector_v5_outcomes_20261013_211324/models"
    cache_path = folder / "validation_predictions.pt"
    selected_name = "outcome_seed_42_snapshot_1.pt"
    selected_path = folder / selected_name
    artifact_path = folder / "candidate_selector_checkpoint.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert cache["split"] == "validation" and cache["test_rows_exported"] == 0
    candidate = cache["candidates"][selected_name]
    assert candidate["sha256"] == sha256(selected_path)
    assert candidate["selection_rule"] == artifact["selection_rule"]
    p, q = cache["calibrated_risks"], candidate["probabilities"]
    labels, mask = cache["labels"].bool(), cache["current_mask"]
    assert len(labels) == cache["samples"] == len(p) == len(q) == len(mask)
    target = labels[:, 0].long() + 2 * labels[:, 1].long()
    counts = torch.bincount(target, minlength=4)
    current_lp = mask == 1
    current_fail = torch.where(current_lp, labels[:, 0], labels[:, 1])
    other_fail = torch.where(current_lp, labels[:, 1], labels[:, 0])
    rescue, harm = current_fail & ~other_fail, ~current_fail & other_fail
    one_fail = labels[:, 0] ^ labels[:, 1]
    marginal = torch.stack((q[:, 1] + q[:, 3], q[:, 2] + q[:, 3]), 1)
    rule = SelectionRule.from_dict(artifact["selection_rule"])
    available = torch.ones(len(labels), dtype=torch.bool)
    desired, warning = propose_outcome_selection(p, q, mask, rule, available, available)
    requested = desired != mask
    gain = torch.where(current_lp, q[:, 1] - q[:, 2], q[:, 2] - q[:, 1])
    q_class = q.argmax(1)
    argmax_action = mask.clone()
    argmax_action[q_class == 1], argmax_action[q_class == 2] = 0, 1
    fixed_gain_action = mask.clone()
    keep_request = requested & (gain > 0.01)
    fixed_gain_action[keep_request] = desired[keep_request]

    def action_metrics(action):
        changed = action != mask
        wrong = torch.where(action == 1, labels[:, 0], labels[:, 1]) & one_fail
        result = {"requests": int(changed.sum())}
        for name, hit, total in (
            ("both_safe_requests", changed & (target == 0), target == 0),
            ("both_failure_requests", changed & (target == 3), target == 3),
            ("correct_rescue", changed & rescue, rescue),
            ("missed_rescue", ~changed & rescue, rescue),
            ("harmful_switch", changed & harm, harm),
            ("dangerous_choice_on_single_failure", wrong, one_fail),
        ):
            count, denominator = int(hit.sum()), int(total.sum())
            result[name] = {"count": count, "denominator": denominator,
                            "rate": fraction(count, denominator)}
        return result

    result = {
        "scope": {
            "model": "OLD LOCAL V5, NOT the server model with 15.55%/49.39%/68.80% results",
            "split": "validation", "rows": len(labels), "test_accessed": False,
            "threshold_search": False, "weights_updated": False,
            "limitations": [
                "Validation was already used for calibration, model selection, and decision fitting; this is not independent generalization evidence.",
                "Cached independent risks and joint probabilities are used; inference is not repeated.",
                "Requests are evaluated before runtime dwell/confirmation/blending and physical closed-loop outcomes.",
                "Both sensors are marked available, matching the offline probability-rule diagnostic.",
                "Argmax and fixed 0.01 gain are predetermined diagnostic comparisons, not accepted deployment rules.",
                "Failure labels indicate 1-second fixed-branch counterfactual outcomes, not all future switching sequences.",
            ],
        },
        "inputs": {
            "cache": str(cache_path), "cache_sha256": sha256(cache_path),
            "cache_fields": ["labels", "current_mask", "calibrated_risks",
                             f"candidates/{selected_name}/probabilities"],
            "selected_model": str(selected_path), "selected_model_sha256": sha256(selected_path),
            "decision_artifact": str(artifact_path), "decision_artifact_sha256": sha256(artifact_path),
            "p_order": ["LP_failure_1s", "VP_failure_1s"],
            "q_order": cache["outcome_order"], "mask_mapping": {"0": "VP", "1": "LP"},
            "class_counts": counts.tolist(), "selection_rule": artifact["selection_rule"],
        },
        "p_vs_q_marginals": {},
        "actions": {
            "existing_rule": action_metrics(desired),
            "argmax_class_action": action_metrics(argmax_action),
            "existing_rule_and_fixed_gain_gt_0_01": action_metrics(fixed_gain_action),
        },
        "argmax_confusion_true_rows_predicted_columns": torch.bincount(4 * target + q_class, minlength=16).reshape(4, 4).tolist(),
    }
    for j, name in enumerate(("LP", "VP")):
        delta = (p[:, j] - marginal[:, j]).abs()
        result["p_vs_q_marginals"][name] = {
            "mean_absolute_difference": float(delta.mean()),
            "p95_absolute_difference": float(torch.quantile(delta, .95)),
            "fraction_absolute_difference_gt_0_1": float((delta > .1).float().mean()),
            "independent_risk_p": binary_metrics(p[:, j], labels[:, j]),
            "joint_q_marginal": binary_metrics(marginal[:, j], labels[:, j]),
        }
    p_diff, q_diff = p[:, 0] - p[:, 1], q[:, 1] - q[:, 2]
    result["single_failure_direction"] = {
        "rows": int(one_fail.sum()),
        "wrong_p_direction": int((((p_diff > 0) != (target == 1)) & one_fail).sum()),
        "wrong_q_direction": int((((q_diff > 0) != (target == 1)) & one_fail).sum()),
        "conditional_q1_over_q1_plus_q2_auroc": binary_auroc(
            (q[:, 1] / (q[:, 1] + q[:, 2]).clamp_min(1e-12))[one_fail], target[one_fail] == 1),
    }
    result["single_failure_event_mass_g_q1_plus_q2"] = binary_metrics(q[:, 1] + q[:, 2], one_fail)
    result["direction_decomposition"] = {
        "g": "q1 + q2 = probability exactly one branch fails",
        "d": "q1 / (q1 + q2) = conditional direction given exactly one branch fails",
        "current_log_ratio": "log(q1/q2) = logit(d), which removes g (except numerical clamping)",
        "absolute_gain_lp_to_vp": "q1 - q2 = g * (2*d - 1)",
    }
    safe_request = requested & (target == 0)
    failed_request = requested & (target == 3)
    result["existing_request_diagnostics"] = {
        "all_requests_q_argmax_class_counts": torch.bincount(q_class[requested], minlength=4).tolist(),
        "both_safe_requests_q_argmax_class_counts": torch.bincount(q_class[safe_request], minlength=4).tolist(),
        "both_failure_requests_q_argmax_class_counts": torch.bincount(q_class[failed_request], minlength=4).tolist(),
        "both_safe_requested_gain_quantiles": {str(v): float(torch.quantile(gain[safe_request], v)) for v in (0., .1, .5, .9, 1.)},
        "both_safe_requests_gain_lt_0_01": int((safe_request & (gain < .01)).sum()),
        "both_safe_requests_gain_lt_0_01_fraction": fraction(int((safe_request & (gain < .01)).sum()), int(safe_request.sum())),
        "fixed_gain_gt_0_01_additional_rescues_blocked": int((requested & rescue & ~keep_request).sum()),
        "fixed_gain_gt_0_01_warning_unchanged": True,
    }
    result["joint_probability_diagnostics"] = {
        "q0": binary_metrics(q[:, 0], target == 0),
        "q3": binary_metrics(q[:, 3], target == 3),
        "product_of_independent_marginals": binary_metrics(p.prod(1), target == 3),
        "q3_exceeds_p_frechet_upper_by_gt_0_05": int((q[:, 3] > p.min(1).values + .05).sum()),
        "q3_below_p_frechet_lower_by_gt_0_05": int((q[:, 3] < (p.sum(1) - 1).clamp_min(0) - .05).sum()),
        "q0_hold_count": int((q[:, 0] >= rule.both_safe_probability_threshold).sum()),
        "warning_at_existing_fixed_threshold": {
            "threshold": rule.both_failure_flag_threshold,
            "flagged": int(warning.sum()), "true_positive": int((warning & (target == 3)).sum()),
            "recall": fraction(int((warning & (target == 3)).sum()), int(counts[3])),
            "precision": fraction(int((warning & (target == 3)).sum()), int(warning.sum())),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
