"""Train-only statistics and explicit joint-selector prediction diagnostics."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .joint_data import JointShardDataset, file_sha256
from .joint_model import FeatureNormalizer, joint_selector_loss, propose_joint_argmax, risks_from_joint
from .metrics import json_safe_metrics, selector_metrics
from .outcome_decision import OutcomeGuardConfig, outcome_evaluation_checks
from .outcomes import OUTCOME_ORDER, fit_outcome_calibration, outcome_classes


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe_metrics(data), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LOSS_REDUCTION = "row_mean_of_fixed_training_normalized_class_weighted_nll"


class GlobalRowBatchSampler(Sampler):
    """A single permutation of ALL source rows; no replacement or shard blocks."""
    def __init__(self, dataset, batch_size, *, shuffle, generator=None):
        if batch_size < 1 or len(dataset) < 1:
            raise ValueError("positive batch size and nonempty dataset required")
        self.dataset, self.batch_size = dataset, int(batch_size)
        self.shuffle, self.generator = bool(shuffle), generator
        self.source_counts = list(getattr(dataset, "source_counts", [len(dataset)]))
        if any(n < 0 for n in self.source_counts) or sum(self.source_counts) != len(dataset):
            raise ValueError("source counts must sum to the dataset length")
        self.epoch_statistics = None

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        order = (torch.randperm(len(self.dataset), generator=self.generator) if self.shuffle
                 else torch.arange(len(self.dataset)))
        boundaries = torch.tensor(self.source_counts, dtype=torch.long).cumsum(0)[:-1]
        sources = torch.bucketize(order, boundaries, right=True)
        number_sources = len(self.source_counts)
        totals, low, high = torch.zeros(number_sources, dtype=torch.long), None, torch.zeros(number_sources, dtype=torch.long)
        stats = {"mode": "global_row_permutation_without_replacement" if self.shuffle else "sequential_rows",
            "rows": 0, "batches": 0, "pure_source_batches": 0, "pure_source_full_batches": 0,
            "source_rows": [0]*number_sources, "complete": False}
        self.epoch_statistics = stats
        for offset in range(0, len(order), self.batch_size):
            rows = order[offset:offset+self.batch_size]
            count = torch.bincount(sources[offset:offset+self.batch_size], minlength=number_sources)
            totals += count
            low = count.clone() if low is None else torch.minimum(low, count)
            high = torch.maximum(high, count)
            pure = int((count > 0).sum()) == 1
            stats.update(rows=stats["rows"]+len(rows), batches=stats["batches"]+1,
                pure_source_batches=stats["pure_source_batches"]+int(pure),
                pure_source_full_batches=stats["pure_source_full_batches"]+int(pure and len(rows)==self.batch_size),
                source_rows=totals.tolist(), source_rows_per_batch_min=low.tolist(), source_rows_per_batch_max=high.tolist())
            yield rows.tolist()
        stats["complete"] = True


def make_loader(dataset, args, device, shuffle=False, seed=0):
    if shuffle and isinstance(dataset, JointShardDataset) and not dataset.random_access_ready:
        raise ValueError("prepare the training row cache before global shuffle")
    return DataLoader(dataset, batch_sampler=GlobalRowBatchSampler(dataset, args.batch_size,
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed)), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)


@torch.no_grad()
def fit_statistics(loader, class_weighting):
    """Only the loader's training rows fit means, stds and class weights."""
    total = torch.zeros(283, dtype=torch.float64)
    square, finite_count = torch.zeros_like(total), torch.zeros_like(total)
    counts, rows, nonfinite = torch.zeros(4, dtype=torch.long), 0, 0
    for batch in loader:
        x = batch["features"].reshape(-1, 283).double()
        valid = torch.isfinite(x)
        safe = torch.where(valid, x, 0.)
        total += safe.sum(0)
        square += safe.square().sum(0)
        finite_count += valid.sum(0)
        nonfinite += int((~valid).sum())
        counts += torch.bincount(outcome_classes(batch["labels"]), minlength=4)
        rows += len(batch["labels"])
    if rows == 0 or (counts == 0).any() or (finite_count == 0).any():
        raise ValueError("training needs all four outcomes and observed features in every column")
    mean = total / finite_count
    std = (square / finite_count - mean.square()).clamp_min(1e-12).sqrt()
    normalizer = FeatureNormalizer(283)
    normalizer.set_statistics(mean.float(), std.float())
    if class_weighting not in ("none", "sqrt_inverse_frequency"):
        raise ValueError("unknown class weighting")
    raw_weights = torch.ones(4, dtype=torch.float64) if class_weighting == "none" else (counts.max()/counts.double()).sqrt()
    # One fixed denominator, fitted on TRAIN only. Never divide by a batch's
    # class-weight sum: pure-class minibatches must retain their relative weight.
    weight_normalizer = (counts.double()*raw_weights).sum()/rows
    weights = (raw_weights/weight_normalizer).float()
    return normalizer, weights, {"fit_split": "train", "rows": rows, "class_counts": counts.tolist(),
                                 "nonfinite_feature_values_imputed": nonfinite, "class_weights": weights.tolist(),
                                 "class_weighting": class_weighting, "normalizer_fit_split": "train",
                                 "raw_class_weights": raw_weights.tolist(),
                                 "class_weight_normalizer": float(weight_normalizer),
                                 "class_weight_normalizer_fit_split": "train",
                                 "train_expected_class_weight": float((counts.double()*weights.double()).sum()/rows),
                                 "ce_reduction": LOSS_REDUCTION}


class RowMeanLossAccumulator:
    """Additive row means; independent of batch partitions for fixed logits.

    During training weights change between batches, so the resulting log is an
    online objective, not a fixed final-model train-set evaluation.
    """
    def __init__(self):
        self.rows = 0
        self.sums = {k: 0. for k in ("joint_ce", "lp_bce", "vp_bce")}

    def update(self, parts, rows):
        if rows < 1 or set(parts) != set(self.sums):
            raise ValueError("positive row count and the three loss components required")
        self.rows += rows
        for name in self.sums:
            self.sums[name] += float(parts[name].detach())*rows

    def result(self, marginal_weight):
        if not self.rows:
            raise ValueError("cannot reduce empty loss history")
        components = {k: v/self.rows for k, v in self.sums.items()}
        total = components["joint_ce"]+marginal_weight*(components["lp_bce"]+components["vp_bce"])/2
        return total, components


@torch.no_grad()
def collect_logits(model, normalizer, loader, device, progress=None):
    model.eval()
    values, labels, masks, ids = [], [], [], []
    for number, batch in enumerate(loader, 1):
        x = batch["features"]
        current = x[:, -1, -2]
        if not torch.isfinite(current).all() or not ((current == 0) | (current == 1)).all():
            raise ValueError("observable current branch must be binary VP=0/LP=1")
        z = model(normalizer(x.to(device, non_blocking=True)))
        if not torch.isfinite(z).all():
            raise RuntimeError("nonfinite joint network logits")
        values.append(z.float().cpu())
        labels.append(batch["labels"].float().cpu())
        masks.append(current.float().cpu())
        ids.append(batch["sample_id"].reshape(-1).cpu())
        if progress and (number == 1 or number % 20 == 0 or number == len(loader)):
            print(f"[joint predict] {progress}: batch={number}/{len(loader)}", flush=True)
    if not values:
        raise ValueError("cannot evaluate an empty split")
    return {"logits": torch.cat(values), "labels": torch.cat(labels),
            "current_mask": torch.cat(masks), "sample_id": torch.cat(ids)}


def validation_objective(predictions, class_weights, marginal_weight):
    with torch.no_grad():
        value, parts = joint_selector_loss(predictions["logits"], predictions["labels"], class_weights, marginal_weight)
    if not torch.isfinite(value):
        raise RuntimeError("nonfinite validation objective")
    return float(value), {k: float(v) for k, v in parts.items()}


def calibrate_q(predictions, temperature, biases, device):
    with torch.no_grad():
        return torch.softmax(predictions["logits"].to(device) / temperature.to(device) + biases.to(device), 1).cpu()


PROBABILITY_MODES = ("raw", "temperature", "temperature_bias")


def fit_probability_profiles(predictions, class_weights):
    """Fit diagnostic calibration on validation only; never choose the primary mode."""
    z = predictions["logits"].detach().cpu().double()
    y = outcome_classes(predictions["labels"].cpu())
    before = float(torch.nn.functional.cross_entropy(z, y))
    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_t], lr=.5, max_iter=150, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z/log_t.clamp(math.log(1e-3), math.log(100.)).exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        t = log_t.clamp(math.log(1e-3), math.log(100.)).exp()
        after = float(torch.nn.functional.cross_entropy(z/t, y))
        if not math.isfinite(after) or after > before:
            t, after = torch.ones_like(t), before
    temperature, biases, info = fit_outcome_calibration(predictions["logits"], predictions["labels"], class_weights)
    return {
        "raw": {"temperature": [1.], "biases": [0.]*4,
            "info": {"method": "identity", "fit_split": None, "uses_test_labels": False}},
        "temperature": {"temperature": t.float().tolist(), "biases": [0.]*4,
            "info": {"method": "multinomial_temperature_only", "fit_split": "validation",
                "uses_test_labels": False, "objective": "unweighted_cross_entropy",
                "before_log_loss": before, "after_log_loss": after}},
        "temperature_bias": {"temperature": temperature.tolist(), "biases": biases.tolist(), "info": info},
    }


def profile_probabilities(predictions, profile, device):
    return calibrate_q(predictions, torch.tensor(profile["temperature"]), torch.tensor(profile["biases"]), device)


@torch.no_grad()
def save_full_predictions(path, predictions, profiles_q, first_steps, dataset, source):
    """Save existing predictions for later audit, without another model forward."""
    path = Path(path)
    y = predictions["labels"].cpu().bool()
    current = predictions["current_mask"].cpu().long()
    target = outcome_classes(y)
    current_failure = torch.where(current == 1, y[:, 0], y[:, 1])
    alternative_failure = torch.where(current == 1, y[:, 1], y[:, 0])
    rescue, harmful = current_failure & ~alternative_failure, ~current_failure & alternative_failure
    arrays = {"logits": predictions["logits"].cpu().numpy(), "labels": y.numpy().astype(np.uint8),
        "current_branch": current.numpy().astype(np.uint8), "sample_id": predictions["sample_id"].cpu().numpy(),
        "split_row_index": np.arange(len(y)), "first_failure_steps": first_steps.numpy().astype(np.int16),
        "shard_index": np.repeat(np.arange(len(dataset.lengths), dtype=np.int32), dataset.lengths),
        "row_in_shard": np.concatenate([np.arange(n, dtype=np.int32) for n in dataset.lengths])}
    for name, q in profiles_q.items():
        q = q.cpu()
        desired, warning = propose_joint_argmax(q, current)
        desired = desired.long()
        predicted, changed = q.argmax(1), desired != current
        selected_failure = torch.where(desired == 1, y[:, 0], y[:, 1])
        missed = rescue & ~changed
        values = {"probabilities": q, "predicted_class": predicted, "desired_branch": desired,
            "switch_request": changed, "warning": warning, "selected_failure": selected_failure,
            "single_failure_wrong": (target == 1) | (target == 2),
            "missed_rescue": missed, "correct_rescue": rescue & changed,
            "harmful_switch": harmful & changed,
            "wrong_safe_hold": missed & (predicted == 0), "wrong_both_failure_hold": missed & (predicted == 3),
            "wrong_direction": ((target == 1) & (predicted == 2)) | ((target == 2) & (predicted == 1))}
        values["single_failure_wrong"] &= selected_failure
        for key, value in values.items():
            array = value.numpy()
            if key in ("predicted_class", "desired_branch"):
                array = array.astype(np.uint8)
            arrays[f"{key}_{name}"] = array
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    metadata = {"format": "selector_joint_predictions", "version": 2, "rows": len(y),
        "full_split_export": True, "prediction_file": str(path.resolve()), "prediction_sha256": file_sha256(path),
        "class_order": list(OUTCOME_ORDER), "label_order": ["LP", "VP"],
        "current_branch_codes": {"VP": 0, "LP": 1}, "source": source,
        "primary_probability_mode": source["decision_probability_mode"],
        "profiles": list(profiles_q), "shard_order": [str(p) for p in dataset.shards],
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "decision_scope": "pointwise_argmax_no_dwell_confirmation_health_or_blending_gates",
        "weights_and_calibration_frozen_for_export": True,
        "notes": ["first_failure_steps are offline audit fields, never network inputs.",
                  "The train split includes the recorded base and all supplement sources in source order."]}
    save_json(path.with_suffix(".json"), metadata)
    return {"predictions": str(path.resolve()), "manifest": str(path.with_suffix('.json').resolve()),
            "rows": len(y), "sha256": metadata["prediction_sha256"], "full_split_export": True}


def ratio(count, samples):
    return count / samples if samples else float("nan")


@torch.no_grad()
def report_joint(q, predictions, max_examples=30):
    y, current = predictions["labels"], predictions["current_mask"]
    p = risks_from_joint(q)
    target, predicted = outcome_classes(y), q.argmax(1)
    desired, warning = propose_joint_argmax(q, current)
    # Reuse existing calibration/AUROC diagnostics, replacing all action fields.
    shared = selector_metrics(p, y, current_mask=current, outcome_probabilities=q)
    metrics = {k: v for k, v in shared.items() if not k.startswith(("selection/", "switch/", "failure/"))}
    lp, vp = y[:, 0].bool(), y[:, 1].bool()
    both_safe, both_failed, one_safe = target == 0, target == 3, lp ^ vp
    changed = desired != current
    selected_failure = torch.where(desired == 1, lp, vp)
    current_failure = torch.where(current == 1, lp, vp)
    alternative_failure = torch.where(current == 1, vp, lp)
    wrong, rescue = one_safe & selected_failure, current_failure & ~alternative_failure
    harmful = ~current_failure & alternative_failure
    metrics.update({
        "selection/dangerous_branch_count": int(wrong.sum()),
        "selection/dangerous_branch_rate": ratio(int(wrong.sum()), int(one_safe.sum())),
        "selection/lp_fraction": float((desired == 1).float().mean()),
        "failure/always_lp": float(lp.float().mean()), "failure/always_vp": float(vp.float().mean()),
        "failure/selector": float(selected_failure.float().mean()), "failure/oracle": float(both_failed.float().mean()),
        "switch/request_count": int(changed.sum()), "switch/request_fraction": float(changed.float().mean()),
        "switch/hold_count": int((~changed).sum()), "switch/hold_fraction": float((~changed).float().mean()),
        "switch/lp_to_vp_count": int((changed & (current == 1)).sum()),
        "switch/vp_to_lp_count": int((changed & (current == 0)).sum()),
        "selection/low_risk_hold_count": int((predicted == 0).sum()),
        "selection/low_risk_hold_true_safe_count": int(((predicted == 0) & both_safe).sum()),
        "selection/both_failure_hold_count": int((predicted == 3).sum()),
        "selection/both_failure_hold_true_positive_count": int(((predicted == 3) & both_failed).sum()),
        "selection/both_failure_hold_single_failure_count": int(((predicted == 3) & one_safe).sum()),
    })
    best_fixed = min(metrics["failure/always_lp"], metrics["failure/always_vp"])
    metrics["failure/improvement_over_best_fixed"] = ((best_fixed-metrics["failure/selector"])/best_fixed if best_fixed else 0.)
    for name, subset in (("lp", target == 1), ("vp", target == 2)):
        count, bad = int(subset.sum()), int((subset & selected_failure).sum())
        metrics[f"selection/{name}_only_failure_wrong_count"] = bad
        metrics[f"selection/{name}_only_failure_wrong_rate"] = ratio(bad, count)
        metrics[f"selection/{name}_only_failure_correct_count"] = count-bad
    for name, subset, denominator in (
        ("missed_rescue", rescue & ~changed, rescue), ("correct_rescue", rescue & changed, rescue),
        ("harmful_switch", harmful & changed, harmful),
        ("unnecessary_switch", ~current_failure & changed, ~current_failure),
        ("both_safe_switch", both_safe & changed, both_safe),
        ("both_failure_switch", both_failed & changed, both_failed),
    ):
        count, samples = int(subset.sum()), int(denominator.sum())
        metrics[f"switch/{name}_count"], metrics[f"switch/{name}_samples"] = count, samples
        metrics[f"switch/{name}_rate"] = ratio(count, samples)
    tp, flagged, positives = int((warning & both_failed).sum()), int(warning.sum()), int(both_failed.sum())
    metrics.update({
        "selection/both_unreliable_flag_count": flagged,
        "selection/both_failure_flag_true_positive_count": tp,
        "selection/both_failure_flag_false_positive_count": flagged-tp,
        "selection/both_failure_flag_false_negative_count": positives-tp,
        "selection/both_failure_flag_recall": ratio(tp, positives),
        "selection/both_unreliable_flag_precision": ratio(tp, flagged),
        "selection/both_failure_flag_single_failure_count": int((warning & one_safe).sum()),
        "selection/both_failure_flag_both_safe_count": int((warning & both_safe).sum()),
    })
    if (metrics["selection/dangerous_branch_count"] != metrics["switch/missed_rescue_count"] + metrics["switch/harmful_switch_count"]
            or ((predicted == 0) & changed).any() or ((predicted == 3) & changed).any()):
        raise RuntimeError("joint selection metric/mapping invariant failed")
    confusion = torch.bincount(target*4+predicted, minlength=16).reshape(4, 4)
    request_cross = torch.bincount((target*4+predicted)*2+changed.long(), minlength=32).reshape(4, 4, 2)

    def examples(mask):
        rows = []
        for i in mask.nonzero().flatten()[:max_examples].tolist():
            rows.append({"split_row_index": i, "sample_id": int(predictions["sample_id"][i]),
                "labels_lp_vp": y[i].int().tolist(), "true_outcome": OUTCOME_ORDER[int(target[i])],
                "predicted_outcome": OUTCOME_ORDER[int(predicted[i])], "outcome_probabilities": q[i].tolist(),
                "derived_risks_lp_vp": p[i].tolist(), "current_mask": int(current[i]),
                "desired_mask": int(desired[i]), "warning": bool(warning[i])})
        return rows

    return {"samples": len(y), "metrics": metrics,
        "diagnostic_checks": outcome_evaluation_checks(metrics, OutcomeGuardConfig()),
        "class_order": list(OUTCOME_ORDER), "class_counts": torch.bincount(target, minlength=4).tolist(),
        "confusion_rows_true_columns_predicted": confusion.tolist(),
        "true_outcome_x_predicted_outcome_x_request": {"request_axis": ["no_request", "request"], "counts": request_cross.tolist()},
        "marginal_consistency_max_error": float((p - risks_from_joint(q)).abs().max()),
        "exact_top_probability_tie_rows": int(((q == q.max(1, keepdim=True).values).sum(1)>1).sum()),
        "examples": {"lp_failure_misclassified_safe": examples((target == 1) & (predicted == 0)),
            "vp_failure_misclassified_safe": examples((target == 2) & (predicted == 0)),
            "missed_rescue": examples(rescue & ~changed), "harmful_switch": examples(harmful & changed),
            "both_failure_missed_warning": examples(both_failed & ~warning)},
        "selection_scope": "pointwise_joint_argmax_using_observed_current_branch",
        "runtime_gates_evaluated": False,
        "notes": ["Masks: VP=0, LP=1; pure argmax; exact ties use the first class.",
                  "The explicit probability profile is applied once; all branch risks are exact marginals.",
                  "Hold metrics refer to predicted classes, not additional probability thresholds.",
                  "Rows may be correlated. Selected failure labels are not physical closed-loop fall rates.",
                  "Diagnostic checks are not deployment acceptance; timing/health/blending gates were not replayed."]}


def threshold_warning_report(q, labels, threshold):
    target, flag = outcome_classes(labels) == 3, q[:, 3] >= threshold
    tp, fp = int((flag & target).sum()), int((flag & ~target).sum())
    return {"threshold": threshold, "true_positives": tp, "false_positives": fp,
            "false_negatives": int(target.sum())-tp, "recall": ratio(tp, int(target.sum())),
            "precision": ratio(tp, tp+fp), "diagnostic_only": True,
            "changes_argmax_actions": False, "changes_primary_argmax_warning": False}
