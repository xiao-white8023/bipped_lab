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
    ranks_sorted = torch.arange(1, len(order) + 1, dtype=torch.float64)
    # Equal scores must receive their average rank; arbitrary ordering of ties
    # otherwise makes AUROC depend on input order.
    _, counts = torch.unique_consecutive(sorted_probability, return_counts=True)
    offset = 0
    for count in counts.tolist():
        end = offset + count
        ranks_sorted[offset:end] = ranks_sorted[offset:end].mean()
        offset = end
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
    if probabilities.shape != labels.shape or probabilities.shape[-1] != 4:
        raise ValueError("probabilities and labels must have matching [samples, 4] shapes")
    names = ("lp_short", "lp_long", "vp_short", "vp_long")
    metrics: dict[str, float] = {}
    for index, name in enumerate(names):
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
        float(dangerous_choice.float().sum() / one_safe.float().sum().clamp_min(1.0))
    )
    metrics["selection/lp_fraction"] = float(choose_lp.float().mean())
    finite_aurocs = [value for key, value in metrics.items() if key.startswith("auroc/") and math.isfinite(value)]
    finite_eces = [value for key, value in metrics.items() if key.startswith("ece/") and math.isfinite(value)]
    metrics["auroc/min"] = min(finite_aurocs) if finite_aurocs else float("nan")
    metrics["ece/max"] = max(finite_eces) if finite_eces else float("nan")
    return metrics
