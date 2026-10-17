"""Fit a new two-risk selector from random initialization on verified clean data.

No selector checkpoint is accepted as an initialization. Train fits weights and
feature statistics; validation selects the seed/epoch and fits calibration.
Test shards are inspected only by the existing source-integrity auditor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.scripts.train_selector import (
    evaluate, fit_normalizer_and_weights, seed_everything,
)
from legged_lab.scripts.train_selector_clean_outcomes import (
    cpu_state, loader_for, save_json, save_torch,
)
from legged_lab.selector.clean_sources import audit_filtered_training_sources
from legged_lab.selector.config import SelectorConfig, TrainingConfig
from legged_lab.selector.feature_contract import dataset_feature_contract
from legged_lab.selector.frozen_policy import dataset_policy_identity, sha256
from legged_lab.selector.labels import dataset_safety_label_contract
from legged_lab.selector.metrics import json_safe_metrics, selector_metrics
from legged_lab.selector.model import (
    SelectorNetwork, calibrate_probabilities, fit_affine_calibration,
    weighted_binary_cross_entropy,
)
from legged_lab.selector.supplement_dataset import SelectorTrainingDataset


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def changed_components(initial, trained):
    changes = {}
    for prefix in ("frame_encoder.", "temporal.", "head."):
        names = [name for name in initial if name.startswith(prefix)]
        changes[prefix.rstrip(".")] = sum(
            not torch.equal(initial[name], trained[name]) for name in names
        )
    if not all(changes.values()):
        raise RuntimeError(f"risk network did not update every component: {changes}")
    return changes


def train_seed(seed, train_dataset, validation_loader, normalizer, positive_weights,
               selector_cfg, args, device):
    seed_everything(seed)
    model = SelectorNetwork(selector_cfg).to(device)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all risk network parameters must be trainable")
    initial = cpu_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = loader_for(train_dataset, args, device, True, seed)
    best_loss, stale, best_epoch = math.inf, 0, None
    best_state, history = None, []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total, samples = 0.0, 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probability = model(normalizer(features))
                loss = weighted_binary_cross_entropy(probability, labels, positive_weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite risk loss, seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(labels)
            samples += len(labels)
        validation_loss, *_ = evaluate(
            model, normalizer, validation_loader, positive_weights, device,
        )
        if not samples or not math.isfinite(validation_loss):
            raise RuntimeError("empty risk training or non-finite validation loss")
        row = {"epoch": epoch, "train_loss": total / samples,
               "validation_loss": validation_loss}
        history.append(row)
        print(f"[selector risks seed={seed}] epoch={epoch:03d} "
              f"train={row['train_loss']:.6f} validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss, stale, best_epoch = validation_loss, 0, epoch
            best_state = cpu_state(model)
            save_torch(args.output / f"risk_seed_{seed}_best.pt", {
                "artifact": "risk_training_snapshot_not_deployable", "seed": seed,
                "best_epoch": best_epoch, "validation_loss": best_loss,
                "model_state_dict": best_state,
            })
        else:
            stale += 1
        save_json(args.output / f"risk_seed_{seed}_history.json", history)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("risk training did not produce a finite checkpoint")
    changes = changed_components(initial, best_state)
    record = {"seed": seed, "best_epoch": best_epoch, "epochs": len(history),
              "validation_loss": best_loss, "initialization": "random",
              "initial_checkpoint": None, "updated_components": changes,
              "initial_state_sha256": state_sha256(initial),
              "trained_state_sha256": state_sha256(best_state),
              "snapshot": str((args.output / f"risk_seed_{seed}_best.pt").resolve())}
    save_json(args.output / f"risk_seed_{seed}_summary.json", record)
    del train_loader
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--supplement-datasets", nargs="+", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (min(args.batch_size, args.max_epochs, args.patience) < 1 or args.num_workers < 0
            or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error("positive sizes, nonnegative workers and distinct uint32 seeds are required")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("learning rate must be positive; weight decay must be nonnegative")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("use a new or empty risk output directory")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate env_tienkung or explicitly use --device cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("[selector risks] RANDOM initialization; encoder, GRU and risk heads all trainable", flush=True)
    audit = audit_filtered_training_sources(args.original_dataset, args.dataset, args.supplement_datasets)
    audit["risk_parent_provenance"] = "risk network fitted from random initialization on these declared training rows"
    audit["normalizer_fit_split"] = "train"
    train_dataset = SelectorTrainingDataset([args.dataset, *args.supplement_datasets])
    validation_dataset = SelectorTrainingDataset([args.dataset], "validation")
    if min(len(train_dataset), len(validation_dataset)) <= 0:
        raise ValueError("nonempty train and validation splits are required")
    if len(train_dataset) != audit["training_samples"]:
        raise RuntimeError("audited training count differs from training loader")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "training_sources.json", audit)
    cfg = SelectorConfig()
    cfg.validate()
    statistics_loader = loader_for(train_dataset, args, device)
    print(f"[selector risks] refitting train-only normalizer on {len(train_dataset)} rows", flush=True)
    normalizer, positive_weights = fit_normalizer_and_weights(statistics_loader, cfg.frame_dim)
    del statistics_loader
    normalizer = normalizer.to(device).eval()
    positive_weights = positive_weights.to(device)
    validation_loader = loader_for(validation_dataset, args, device)
    candidates = [train_seed(seed, train_dataset, validation_loader, normalizer,
                             positive_weights, cfg, args, device) for seed in args.seeds]
    chosen = min(candidates, key=lambda record: (record["validation_loss"], record["seed"]))
    snapshot = torch.load(chosen["snapshot"], map_location="cpu", weights_only=False)
    model = SelectorNetwork(cfg).to(device)
    model.load_state_dict(snapshot["model_state_dict"], strict=True)
    _, raw_validation, labels, masks = evaluate(
        model, normalizer, validation_loader, positive_weights, device,
    )
    print(f"[selector risks] selected seed={chosen['seed']}; fitting validation-only calibration", flush=True)
    temperatures, biases, calibration = fit_affine_calibration(
        raw_validation.to(device), labels.to(device), positive_weights=positive_weights,
    )
    temperatures, biases = temperatures.cpu(), biases.cpu()
    validation_metrics = selector_metrics(
        calibrate_probabilities(raw_validation, temperatures, biases), labels, current_mask=masks,
    )
    diagnostic_loader = loader_for(train_dataset, args, device)
    _, raw_train, train_labels, train_masks = evaluate(
        model, normalizer, diagnostic_loader, positive_weights, device,
    )
    train_metrics = selector_metrics(
        calibrate_probabilities(raw_train, temperatures, biases), train_labels,
        current_mask=train_masks,
    )
    del diagnostic_loader
    provenance = {"initialization": "random", "initial_checkpoint": None,
                  "risk_network_retrained": True, "updated_components": chosen["updated_components"],
                  "initial_state_sha256": chosen["initial_state_sha256"],
                  "trained_state_sha256": chosen["trained_state_sha256"],
                  "weights_fit_split": "train", "normalizer_fit_split": "train",
                  "calibration_fit_split": "validation", "selection_split": "validation",
                  "test_used_for_selection": False,
                  "training_samples": len(train_dataset), "selected_seed": chosen["seed"]}
    training_cfg = TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, max_epochs=args.max_epochs, patience=args.patience,
        seeds=tuple(args.seeds))
    artifact = {"version": 3, "seed": chosen["seed"], "selector_config": asdict(cfg),
                "training_config": asdict(training_cfg),
                "frozen_policy": dataset_policy_identity(train_dataset.manifest),
                "safety_label_contract": dataset_safety_label_contract(train_dataset.manifest),
                "selector_feature_contract": dataset_feature_contract(train_dataset.manifest),
                "model_state_dict": cpu_state(model),
                "normalizer_state_dict": cpu_state(normalizer),
                "temperatures": temperatures, "calibration_biases": biases,
                "calibration": calibration, "positive_weights": positive_weights.cpu(),
                "data_sources": audit, "risk_training": provenance,
                "metrics": {f"validation/{key}": value for key, value in validation_metrics.items()}}
    checkpoint = args.output / "retrained_risk_checkpoint.pt"
    save_torch(checkpoint, artifact)
    summary = {"report_version": 1, "completed": True,
               "network_training": "randomly initialized risk encoder, GRU and two risk heads",
               "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
               "risk_training": provenance, "selected": chosen, "all_candidates": candidates,
               "train_metrics": train_metrics, "validation_metrics": validation_metrics,
               "calibration": calibration, "training_sources": audit,
               "test_used_for_selection": False,
               "notes": ["No prior selector weights or normalizer are loaded.",
                         "Train metrics use the validation-fitted calibration for diagnosis only.",
                         "Test shard labels are inspected by integrity checks, never fitted or scored here.",
                         "This checkpoint initializes stage two; deployment acceptance is decided there."]}
    save_json(args.output / "risk_training_report.json", summary)
    print(json.dumps(json_safe_metrics({"checkpoint": str(checkpoint),
        "seed": chosen["seed"], "best_epoch": chosen["best_epoch"],
        "risk_network_retrained": True, "updated_components": chosen["updated_components"]}),
        ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()