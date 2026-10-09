# 中文导读
# 文件用途：计算四个风险输出的区分度、校准误差，以及离线分支选择的失败指标。
# 使用位置：train_selector.py 记录验证/测试指标并选模型；eval_selector.py 生成总体及分组报告。
# 说明：离线选择使用风险加权和 VP 偏好；此处没有模拟在线状态机的驻留、迟滞或动作过渡。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - binary_auroc：用排序及并列分数平均秩计算二分类 AUROC，单一标签类别时返回 NaN。
# - expected_calibration_error：按概率分箱计算预测概率与实际失败率的加权差异。
# - selector_metrics：汇总四输出 AUROC/ECE、固定分支和 oracle 失败率及危险分支选择率。
# 中文导读结束

"""Dependency-free probability and branch-selection metrics."""

from __future__ import annotations

import math

import torch


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
def selector_metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if (probabilities.ndim != 2 or probabilities.shape != labels.shape
            or probabilities.shape[1] != 4 or probabilities.shape[0] == 0):
        raise ValueError("probabilities and labels must have matching [samples, 4] shapes")
    if (not torch.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("risk probabilities must be finite and in [0, 1]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("risk labels must be binary")
    names = ("lp_short", "lp_long", "vp_short", "vp_long")
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
    lp_risk = 0.65 * probabilities[:, 0] + 0.35 * probabilities[:, 1]
    vp_risk = 0.65 * probabilities[:, 2] + 0.35 * probabilities[:, 3]
    choose_lp = lp_risk + 0.05 < vp_risk
    lp_failure = labels[:, 1].bool()
    vp_failure = labels[:, 3].bool()
    selected_failure = torch.where(choose_lp, lp_failure, vp_failure)
    oracle_failure = lp_failure & vp_failure
    one_safe = lp_failure ^ vp_failure
    dangerous_choice = one_safe & torch.where(choose_lp, lp_failure, vp_failure)
    metrics["coverage/lp_only_failure"] = int((lp_failure & ~vp_failure).sum())
    metrics["coverage/vp_only_failure"] = int((~lp_failure & vp_failure).sum())
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
    metrics["selection/lp_fraction"] = float(choose_lp.float().mean())
    finite_aurocs = [value for key, value in metrics.items() if key.startswith("auroc/") and math.isfinite(value)]
    finite_eces = [value for key, value in metrics.items() if key.startswith("ece/") and math.isfinite(value)]
    metrics["auroc/min"] = min(finite_aurocs) if len(finite_aurocs) == 4 else float("nan")
    metrics["ece/max"] = max(finite_eces) if len(finite_eces) == 4 else float("nan")
    return metrics


def evaluation_checks(
    metrics: dict[str, float], *, min_class_samples: int = 20, min_contrast_samples: int = 50
) -> dict[str, bool]:
    """Require measurable heads and both branch contrasts before passing gates.

    Counts are minimum coverage checks, not statistical confidence guarantees.
    """
    if min_class_samples < 1 or min_contrast_samples < 1:
        raise ValueError("evaluation coverage thresholds must be positive")
    names = ("lp_short", "lp_long", "vp_short", "vp_long")
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

