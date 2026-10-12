"""Train the independent supervised LP/VP risk selector.

Example:
    python legged_lab/scripts/train_selector.py \
        --dataset outputs/g1_renet_selector/dataset \
        --output outputs/g1_renet_selector/models
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.selector.config import SelectorConfig, TrainingConfig  # noqa: E402
from legged_lab.selector.dataset import ShardBatchSampler, SelectorShardDataset, verify_dataset_no_group_leak  # noqa: E402
from legged_lab.selector.metrics import json_safe_metrics, selector_metrics  # noqa: E402
from legged_lab.selector.frozen_policy import dataset_policy_identity  # noqa: E402
from legged_lab.selector.labels import dataset_safety_label_contract  # noqa: E402
from legged_lab.selector.feature_contract import dataset_feature_contract  # noqa: E402
from legged_lab.selector.supplement_dataset import SelectorTrainingDataset, audit_training_sources  # noqa: E402
from legged_lab.selector.model import (  # noqa: E402
    CalibratedSelector,
    FeatureNormalizer,
    SelectorNetwork,
    fit_temperatures,
    weighted_binary_cross_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the G1 RENet LP/VP selector.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--supplement-datasets", nargs="+", type=Path, default=[],
                        help="Training-only supplements; original validation/test stay fixed.")
    parser.add_argument("--stress-datasets", nargs="+", type=Path, default=[],
                        help="Independent supplement validation/test roots, reported separately.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: SelectorShardDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=ShardBatchSampler(dataset, batch_size, shuffle=shuffle),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def fit_normalizer_and_weights(
    loader: DataLoader,
    frame_dim: int,
) -> tuple[FeatureNormalizer, torch.Tensor]:
    total = torch.zeros(frame_dim, dtype=torch.float64)
    total_square = torch.zeros(frame_dim, dtype=torch.float64)
    count = torch.zeros(frame_dim, dtype=torch.float64)
    positive = torch.zeros(4, dtype=torch.float64)
    samples = 0
    for batch in loader:
        features = batch["features"].reshape(-1, frame_dim).double()
        finite = torch.isfinite(features)
        safe = torch.where(finite, features, torch.zeros_like(features))
        total += safe.sum(dim=0)
        total_square += safe.square().sum(dim=0)
        count += finite.sum(dim=0)
        labels = batch["labels"].double()
        positive += labels.sum(dim=0)
        samples += labels.shape[0]
    count = count.clamp_min(1.0)
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1.0e-12)
    normalizer = FeatureNormalizer(frame_dim)
    normalizer.set_statistics(mean.float(), variance.sqrt().float())
    negative = samples - positive
    invalid_heads = (positive == 0) | (negative == 0)
    if invalid_heads.any():
        names = ("lp_short", "lp_long", "vp_short", "vp_long")
        missing = [names[i] for i in invalid_heads.nonzero().flatten().tolist()]
        raise ValueError(
            f"training labels need both safe and failed samples for {missing}; "
            "recollect data with more terrain and fault coverage"
        )
    weights = (negative / positive.clamp_min(1.0)).clamp(1.0, 100.0).float()
    return normalizer, weights


@torch.no_grad()
def evaluate(
    model: SelectorNetwork,
    normalizer: FeatureNormalizer,
    loader: DataLoader,
    positive_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probabilities = []
    labels = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target = batch["labels"].to(device, non_blocking=True)
        probability = model(normalizer(features))
        loss = weighted_binary_cross_entropy(probability, target, positive_weights)
        total_loss += float(loss) * features.shape[0]
        total_samples += features.shape[0]
        probabilities.append(probability.cpu())
        labels.append(target.cpu())
    return total_loss / max(total_samples, 1), torch.cat(probabilities), torch.cat(labels)


def train_seed(
    seed: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    normalizer: FeatureNormalizer,
    positive_weights: torch.Tensor,
    selector_cfg: SelectorConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    output: Path,
    source_audit=None,
    stress_loaders=(),
) -> tuple[Path, dict[str, float]]:
    seed_everything(seed)
    model = SelectorNetwork(selector_cfg).to(device)
    normalizer = copy.deepcopy(normalizer).to(device).eval()
    positive_weights = positive_weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = model(normalizer(features))
                loss = weighted_binary_cross_entropy(probabilities, labels, positive_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * features.shape[0]
            epoch_samples += features.shape[0]
        validation_loss, _, _ = evaluate(
            model, normalizer, validation_loader, positive_weights, device
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(
            f"[selector seed={seed}] epoch={epoch + 1:03d} "
            f"train={record['train_loss']:.6f} validation={validation_loss:.6f}"
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("selector training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_loss, validation_probability, validation_labels = evaluate(
        model, normalizer, validation_loader, positive_weights, device
    )
    temperatures = fit_temperatures(
        validation_probability.to(device), validation_labels.to(device)
    ).cpu()
    calibrated = CalibratedSelector(model, normalizer, temperatures.to(device)).to(device).eval()
    split_metrics: dict[str, float] = {}
    for split_name, loader in (("validation", validation_loader), ("test", test_loader), *stress_loaders):
        split_probabilities = []
        split_labels = []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device, non_blocking=True)
                split_probabilities.append(calibrated(features).cpu())
                split_labels.append(batch["labels"].float())
        measured = selector_metrics(
            torch.cat(split_probabilities), torch.cat(split_labels)
        )
        split_metrics.update(
            {f"{split_name}/{name}": value for name, value in measured.items()}
        )
    metrics = split_metrics
    metrics["validation/loss"] = validation_loss
    metrics["training/epochs"] = float(len(history))
    artifact = {
        "version": 2,
        "seed": seed,
        "selector_config": asdict(selector_cfg),
        "training_config": asdict(train_cfg),
        "frozen_policy": dataset_policy_identity(train_loader.dataset.manifest),
        "safety_label_contract": dataset_safety_label_contract(train_loader.dataset.manifest),
        "selector_feature_contract": dataset_feature_contract(train_loader.dataset.manifest),
        "model_state_dict": model.cpu().state_dict(),
        "normalizer_state_dict": normalizer.cpu().state_dict(),
        "temperatures": temperatures,
        "positive_weights": positive_weights.cpu(),
        "metrics": metrics,
        "history": history,
        "data_sources": source_audit,
    }
    path = output / f"selector_seed_{seed}.pt"
    torch.save(artifact, path)
    (output / f"selector_seed_{seed}.json").write_text(
        json.dumps(json_safe_metrics(metrics), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return path, metrics


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_audit = audit_training_sources(args.dataset, args.supplement_datasets, args.stress_datasets)
    print(f"[selector] source isolation passed; training_samples={source_audit['training_samples']}")
    (args.output / "training_sources.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selector_cfg = SelectorConfig()
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    train_dataset = SelectorTrainingDataset([args.dataset, *args.supplement_datasets])
    validation_dataset = SelectorTrainingDataset([args.dataset], "validation")
    test_dataset = SelectorTrainingDataset([args.dataset], "test")
    frozen_metadata = dataset_policy_identity(train_dataset.manifest)
    dataset_safety_label_contract(train_dataset.manifest)
    dataset_feature_contract(train_dataset.manifest)
    print(f"[selector] frozen RENet iteration={frozen_metadata['source_iteration']}; "
          f"VP frames={frozen_metadata['vp_proprio_history_frames']}")
    if min(len(train_dataset), len(validation_dataset), len(test_dataset)) == 0:
        raise RuntimeError("all three dataset splits must contain samples")
    statistics_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=False, device=device
    )
    normalizer, positive_weights = fit_normalizer_and_weights(
        statistics_loader, selector_cfg.frame_dim
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True, device=device
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, args.num_workers, shuffle=False, device=device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, args.num_workers, shuffle=False, device=device
    )
    stress_loaders = []
    for number, root in enumerate(args.stress_datasets):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        split = manifest["metadata"]["supplement_source"]["split"]
        if split == "train":
            raise ValueError("stress-datasets may contain only validation or test supplements")
        dataset = SelectorTrainingDataset([root], split)
        if not len(dataset):
            raise ValueError(f"empty stress dataset: {root}")
        stress_loaders.append((f"stress_{number}_{split}", make_loader(
            dataset, args.batch_size, args.num_workers, shuffle=False, device=device)))
    results = []
    for seed in train_cfg.seeds:
        results.append(
            train_seed(
                seed,
                train_loader,
                validation_loader,
                test_loader,
                normalizer,
                positive_weights,
                selector_cfg,
                train_cfg,
                device,
                args.output,
                source_audit,
                stress_loaders,
            )
        )
    results.sort(
        key=lambda item: (
            (item[1]["validation/selection/dangerous_branch_rate"]
             if math.isfinite(item[1]["validation/selection/dangerous_branch_rate"])
             else float("inf")),
            item[1]["validation/loss"],
        )
    )
    best_path, best_metrics = results[0]
    final_path = args.output / "lp_vp_selector_checkpoint.pt"
    shutil.copy2(best_path, final_path)
    summary = {
        "selected": best_path.name,
        "final_artifact": final_path.name,
        "metrics": best_metrics,
        "all_candidates": {path.name: metrics for path, metrics in results},
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(json_safe_metrics(summary), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(f"[selector] selected {best_path.name}; wrote {final_path}")


if __name__ == "__main__":
    main()
