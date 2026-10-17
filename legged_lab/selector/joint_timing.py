"""Offline first-failure diagnostics; these fields never enter the network."""
from __future__ import annotations

import torch

from .joint_model import propose_joint_argmax
from .outcomes import OUTCOME_ORDER, outcome_classes


def collect_failure_steps(dataset, predictions):
    """Read audit columns in prediction order, separately from the feature loader."""
    steps, labels, ids = [], [], []
    for path, count in zip(dataset.shards, dataset.lengths):
        before = path.stat()
        expected = dataset.shard_stats[str(path)]
        if [before.st_size, before.st_mtime_ns] != expected:
            raise RuntimeError(f"source shard changed before timing diagnostics: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        future = data.get("future_metrics")
        if future is None or future.ndim != 3 or future.shape[:2] != (count, 2):
            raise ValueError(f"timing diagnostics require future_metrics [rows,2,metrics]: {path}")
        first = future[:, :, 0]
        if not torch.isfinite(first).all() or not torch.equal(first, first.round()):
            raise ValueError(f"first failure steps must be finite integers: {path}")
        y = data["labels"].float()
        valid = torch.where(y.bool(), (first >= 1) & (first <= 50), first == -1)
        if not valid.all():
            raise ValueError(f"failure timing and binary labels disagree: {path}")
        steps.append(first.long().clone())
        labels.append(y)
        ids.append(data["scenario_seed"].reshape(-1).long())
        after = path.stat()
        if [after.st_size, after.st_mtime_ns] != expected:
            raise RuntimeError(f"source shard changed during timing diagnostics: {path}")
    if (not steps or not torch.equal(torch.cat(ids), predictions["sample_id"].cpu())
            or not torch.equal(torch.cat(labels), predictions["labels"].cpu())):
        raise RuntimeError("timing audit rows do not match prediction IDs/labels")
    return torch.cat(steps)


@torch.no_grad()
def report_failure_timing(q, predictions, first_steps):
    """Separate classification, safe-branch choice and rescue for each time bin."""
    q = q.cpu()
    target = outcome_classes(predictions["labels"].cpu())
    current = predictions["current_mask"].cpu()
    predicted = q.argmax(1)
    desired, _ = propose_joint_argmax(q, current)
    first_steps = first_steps.cpu()
    if first_steps.shape != (len(target), 2):
        raise ValueError("timing diagnostics need [rows,2] first failure steps")
    bins = ((1, 10), (11, 25), (26, 35), (36, 50))
    rows = []
    for class_id in (1, 2, 3):
        times = first_steps[:, class_id-1] if class_id != 3 else first_steps.max(1).values
        for lo, hi in (*bins, (1, 50)):
            mask = (target == class_id) & (times >= lo) & (times <= hi)
            n = int(mask.sum())
            histogram = torch.bincount(predicted[mask], minlength=4).tolist()
            row = {"class": class_id, "class_name": OUTCOME_ORDER[class_id],
                "interval_s": [round((lo-1)*.02, 2), round(hi*.02, 2)],
                "interval": "left_open_right_closed", "all_times": (lo, hi) == (1, 50),
                "samples": n, "predicted_class_counts": histogram,
                "class_true_positive": histogram[class_id],
                "class_recall": histogram[class_id]/n if n else None}
            if class_id != 3:
                safe = 0 if class_id == 1 else 1
                need = mask & (current != safe)
                need_n = int(need.sum())
                rescued = int((need & (desired == safe)).sum())
                correct = int((mask & (desired == safe)).sum())
                row.update(safe_branch_selected=correct, safe_selection_rate=correct/n if n else None,
                    rescue_needed=need_n, correct_rescue=rescued,
                    rescue_rate=rescued/need_n if need_n else None, missed_rescue=need_n-rescued,
                    q0_missed_rescue=int((need & (predicted == 0)).sum()),
                    q3_missed_rescue=int((need & (predicted == 3)).sum()))
            else:
                requests = int((mask & (desired != current)).sum())
                row.update(switch_requests=requests, switch_request_rate=requests/n if n else None)
            rows.append(row)
    return {"time_origin": "candidate snapshot", "control_dt_s": .02,
        "single_failure_time": "first failure of the failing branch",
        "both_failure_time": "later of the two counterfactual first failures",
        "class_order": list(OUTCOME_ORDER), "rows": rows,
        "scope": "offline pointwise argmax; runtime gates not evaluated; audit times are not model inputs"}
