"""Train a temporal joint selector from scratch with explicit probability profiles."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.selector.joint_data import (
    JointShardDataset, assert_inputs_unchanged, audit_sources, file_sha256,
)
from legged_lab.selector.joint_model import (
    ARCHITECTURES, ARTIFACT_KIND, ARTIFACT_VERSION, JointOutcomeNetwork, JointSelectorConfig,
    changed_components, cpu_state, joint_selector_loss, state_sha256,
)
from legged_lab.selector.joint_training import (
    LOSS_REDUCTION, PROBABILITY_MODES, RowMeanLossAccumulator, collect_logits, fit_probability_profiles,
    fit_statistics, make_loader, profile_probabilities, report_joint, save_full_predictions,
    save_json, save_torch, seed_everything, threshold_warning_report, validation_objective,
)
from legged_lab.selector.joint_timing import collect_failure_steps, report_failure_timing
from legged_lab.selector.frozen_policy import dataset_policy_identity
from legged_lab.selector.labels import dataset_safety_label_contract
from legged_lab.selector.outcomes import OUTCOME_ORDER, outcome_classes
from legged_lab.selector.outcome_decision import OutcomeGuardConfig, fit_joint_warning


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="Base dataset; validation/test come only from it.")
    p.add_argument("--original-dataset", type=Path, help="Required only when the base is an offline-filtered dataset.")
    p.add_argument("--supplement-datasets", type=Path, nargs="+", default=[], help="Any number of independent train-only directed banks.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--history-frames", type=int, choices=(5, 10, 20), help="Default: actual stored length; a shorter real-history view is allowed.")
    p.add_argument("--architecture", choices=ARCHITECTURES, default="branch_gru")
    p.add_argument("--gru-layers", type=int, help="Default: 1 per branch for branch_gru; 3 for the single-stream controls.")
    p.add_argument("--gru-dropout", type=float, default=.1)
    p.add_argument("--marginal-weight", type=float, default=1., help="0=CE-only control; 1=CE+(LP BCE+VP BCE)/2.")
    p.add_argument("--class-weighting", choices=("none", "sqrt_inverse_frequency"), default="sqrt_inverse_frequency")
    p.add_argument("--decision-probability-mode", choices=PROBABILITY_MODES, default="raw",
                   help="Explicit primary profile, fixed before test. Other profiles are diagnostics.")
    p.add_argument("--prediction-splits", choices=("train", "validation", "test"), nargs="*", default=["train", "validation"],
                   help="Save full per-row probabilities and failure times for these splits; no values disables export.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--training-cache-dir", type=Path, help="New directory for read-only random-access arrays; default: output/training_row_cache.")
    p.add_argument("--max-examples", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    if args.gru_layers is None:
        args.gru_layers = 1 if args.architecture == "branch_gru" else 3
    if (min(args.gru_layers, args.max_epochs, args.patience, args.batch_size) < 1 or args.num_workers < 0
            or args.max_examples < 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed > 2**32-1 for seed in args.seeds)
            or len(set(args.prediction_splits)) != len(args.prediction_splits)):
        p.error("invalid layer/epoch/batch/worker/seed/example settings")
    if (not math.isfinite(args.marginal_weight) or args.marginal_weight < 0
            or not math.isfinite(args.gru_dropout) or not 0 <= args.gru_dropout < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        p.error("loss/optimizer settings must be finite and valid")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        p.error("choose a new or empty output directory")
    if args.training_cache_dir is not None and args.training_cache_dir.exists():
        p.error("choose a new training cache directory")
    return args


def train_seed(seed, dataset, validation_loader, normalizer, weights, cfg, args, device):
    seed_everything(seed)
    model = JointOutcomeNetwork(cfg).to(device)
    initial = cpu_state(model)
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("all joint network parameters must be trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = make_loader(dataset, args, device, shuffle=True, seed=seed)
    w = weights.to(device)
    history, best_state, best_value, stale, best_epoch = [], None, math.inf, 0, None
    for epoch in range(1, args.max_epochs+1):
        model.train()
        losses = RowMeanLossAccumulator()
        class_low, class_high, class_rows = None, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
        pure_class_batches, pure_class_full_batches = 0, 0
        for batch_number, batch in enumerate(loader, 1):
            features, labels = batch["features"].to(device, non_blocking=True), batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                z = model(normalizer(features))
                loss, parts = joint_selector_loss(z, labels, w, args.marginal_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss: seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            losses.update(parts, count)
            class_count = torch.bincount(outcome_classes(batch["labels"]), minlength=4)
            class_rows += class_count
            class_low = class_count.clone() if class_low is None else torch.minimum(class_low, class_count)
            class_high = torch.maximum(class_high, class_count)
            pure = int((class_count > 0).sum()) == 1
            pure_class_batches += int(pure)
            pure_class_full_batches += int(pure and count == args.batch_size)
            if batch_number == 1 or batch_number % 50 == 0 or batch_number == len(loader):
                print(f"[joint train seed={seed}] epoch={epoch} batch={batch_number}/{len(loader)} "
                      f"loss={float(loss.detach()):.5f} grad_norm={float(norm):.3f}", flush=True)
        if losses.rows != len(dataset):
            raise RuntimeError("training loader did not visit exactly all declared training rows")
        online, online_components = losses.result(args.marginal_weight)
        prediction = collect_logits(model, normalizer, validation_loader, device)
        validation, components = validation_objective(prediction, weights, args.marginal_weight)
        record = {"epoch": epoch, "train_online_objective": online,
                  "train_online_components": online_components,
                  "train_online_scope": "pre_update_predictions_at_changing_parameters_and_training_dropout",
                  "ce_reduction": LOSS_REDUCTION,
                  "validation_objective": validation, "validation_components": components,
                  "validation_scope": "fixed_epoch_end_model_eval_mode",
                  "train_batch_composition": {**loader.batch_sampler.epoch_statistics,
                      "class_rows": class_rows.tolist(), "class_rows_per_batch_min": class_low.tolist(),
                      "class_rows_per_batch_max": class_high.tolist(), "pure_class_batches": pure_class_batches,
                      "pure_class_full_batches": pure_class_full_batches}}
        history.append(record)
        print(f"[joint train seed={seed}] epoch={epoch:03d} train_online={online:.6f} "
              f"validation={validation:.6f} ce={components['joint_ce']:.6f} "
              f"lp_bce={components['lp_bce']:.6f} vp_bce={components['vp_bce']:.6f} "
              f"pure_class_full_batches={pure_class_full_batches}", flush=True)
        if validation < best_value-1e-6:
            best_value, best_epoch, stale = validation, epoch, 0
            best_state = cpu_state(model)
            save_torch(args.output/"training_snapshots"/f"seed_{seed}_best.pt", {
                "artifact_kind": "joint_training_snapshot_not_deployable", "joint_config": asdict(cfg),
                "model_state_dict": best_state, "seed": seed, "best_epoch": best_epoch,
                "validation_objective": best_value, "initial_state_sha256": state_sha256(initial)})
        else:
            stale += 1
        save_json(args.output/f"seed_{seed}_history.json", history)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite joint training checkpoint")
    proof = {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
             "validation_objective": best_value, "initialization": "random", "initial_checkpoint": None,
             "initial_state_sha256": state_sha256(initial), "trained_state_sha256": state_sha256(best_state),
             "updated_components": changed_components(initial, best_state, cfg.gru_layers),
             "all_network_parameters_trainable": True, "architecture": model.architecture_description()}
    del loader, model, optimizer
    return best_state, proof


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: activate env_tienkung, or use --device cpu explicitly")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("[joint train] auditing actual rows, split isolation and directed bank provenance", flush=True)
    indexes, original, audit = audit_sources(args.dataset, args.supplement_datasets, args.original_dataset)
    frames = args.history_frames or indexes[0]["stored_frames"]
    cfg = JointSelectorConfig(history_frames=frames, architecture=args.architecture,
                              gru_layers=args.gru_layers, gru_dropout=args.gru_dropout)
    cfg.validate()
    train = JointShardDataset(indexes, "train", frames)
    validation = JointShardDataset([indexes[0]], "validation", frames)
    test = JointShardDataset([indexes[0]], "test", frames)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("nonempty train/validation/test splits required")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/"training_sources.json", audit)
    print(f"[joint train] architecture={cfg.architecture}, frames={frames}, GRU layers per stream={cfg.gru_layers}; "
          f"rows train={len(train)} validation={len(validation)} test={len(test)}", flush=True)
    statistics_loader = make_loader(train, args, device)
    normalizer, weights, statistics = fit_statistics(statistics_loader, args.class_weighting)
    del statistics_loader
    if statistics["class_counts"] != audit["train_class_counts"] or statistics["rows"] != audit["training_samples"]:
        raise RuntimeError("training statistics and source audit disagree")
    save_json(args.output/"training_statistics.json", statistics)
    cache = train.prepare_random_access(args.training_cache_dir or args.output/"training_row_cache")
    save_json(args.output/"training_cache.json", cache)
    sampler_config = {"mode": "global_row_permutation_without_replacement",
        "scope": "all_base_and_supplement_training_rows", "class_balanced_resampling": False,
        "every_row_once_per_epoch": True, "batch_size": args.batch_size,
        "source_roots": train.source_roots, "source_counts": train.source_counts}
    loss_config = {"marginal_weight": args.marginal_weight, "class_weighting": args.class_weighting,
        "class_weights": weights.tolist(), "raw_class_weights": statistics["raw_class_weights"],
        "class_weight_normalizer": statistics["class_weight_normalizer"],
        "class_weight_normalizer_fit_split": "train", "ce_reduction": LOSS_REDUCTION,
        "bce_reduction": "row_mean_then_mean_of_LP_and_VP", "bce_positive_weights": None}
    print(f"[joint train] global row shuffle; fixed train weight normalizer={statistics['class_weight_normalizer']:.6f}; "
          f"cache={cache['payload_bytes']/1024**3:.2f} GiB", flush=True)
    normalizer = normalizer.to(device).eval()
    validation_loader = make_loader(validation, args, device)
    best_state, selected, candidates = None, None, []
    for seed in args.seeds:
        state, proof = train_seed(seed, train, validation_loader, normalizer, weights, cfg, args, device)
        candidates.append(proof)
        if selected is None or (proof["validation_objective"], seed) < (selected["validation_objective"], selected["seed"]):
            best_state, selected = state, proof
    model = JointOutcomeNetwork(cfg).to(device)
    model.load_state_dict(best_state, strict=True)
    raw_validation = collect_logits(model, normalizer, validation_loader, device, "selected validation")
    print(f"[joint train] primary={args.decision_probability_mode}; fitting diagnostic profiles on validation only", flush=True)
    profiles = fit_probability_profiles(raw_validation, weights)
    validation_q = {name: profile_probabilities(raw_validation, profile, device) for name, profile in profiles.items()}
    for name, profile in profiles.items():
        _, warning = fit_joint_warning(validation_q[name][:, 3], outcome_classes(raw_validation["labels"]) == 3, OutcomeGuardConfig())
        profile["warning_threshold_diagnostic"] = {**warning, "probability_mode": name}
    primary = profiles[args.decision_probability_mode]
    temperature, biases = torch.tensor(primary["temperature"]), torch.tensor(primary["biases"])
    calibration, warning_fit = primary["info"], primary["warning_threshold_diagnostic"]
    architecture = model.architecture_description()
    implementation = "global_mix_fixed_weight_mean_v4_architecture_profiles"
    checkpoint = args.output/"joint_selector_checkpoint.pt"
    artifact = {"version": ARTIFACT_VERSION, "artifact_kind": ARTIFACT_KIND, "joint_config": asdict(cfg),
        "model_state_dict": best_state, "normalizer_state_dict": cpu_state(normalizer),
        "joint_temperature": temperature, "joint_biases": biases, "outcome_order": list(OUTCOME_ORDER),
        "architecture": architecture, "decision_probability_mode": args.decision_probability_mode,
        "probability_mode_selection": "explicit_run_configuration_before_test", "calibration_candidates": profiles,
        "frozen_policy": dataset_policy_identity(indexes[0]["manifest"]),
        "safety_label_contract": dataset_safety_label_contract(indexes[0]["manifest"]),
        "source_feature_contract": indexes[0]["feature_contract"], "model_input_shape": [frames, 283],
        "training_implementation": implementation,
        "loss_config": loss_config, "sampler_config": sampler_config,
        "decision_policy": {"mode": "joint_argmax", "safe": "hold", "both_failure": "hold_and_warn",
                            "lp_only_failure": "VP", "vp_only_failure": "LP", "probability_mode": args.decision_probability_mode},
        "statistics": statistics, "calibration": calibration, "training_sources": audit,
        "selected_training": selected, "all_candidates": candidates, "initial_checkpoint": None,
        "selection_fit_split": "validation", "test_used_for_selection": False,
        "warning_threshold_diagnostic": warning_fit, "deployment_ready": False,
        "runtime_gates_evaluated": False,
        "notes": ["All configured encoders, GRUs, attention readouts and the joint head were trained from scratch.",
                  "The frozen RENet used to generate dataset features/labels remains unchanged.",
                  "V7 records the architecture explicitly; existing V6 single-stream artifacts remain loadable.",
                  "Failure times are audit-only and never network inputs; runtime gates were not replayed."]}
    save_torch(checkpoint, artifact)
    digest = file_sha256(checkpoint)
    freeze = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256_before_test": digest,
        "selected": selected, "model_input_shape": [frames, 283], "calibration_fit_split": calibration["fit_split"],
        "diagnostic_calibration_fit_split": "validation", "decision_probability_mode": args.decision_probability_mode,
        "probability_mode_selection": "explicit_run_configuration_before_test", "architecture": architecture,
        "selection_objective": "raw_validation_training_class_weighted_CE_plus_marginal_BCE",
        "loss_config": loss_config, "sampler_config": sampler_config,
        "test_used_for_selection": False, "decision_policy": artifact["decision_policy"],
        "warning_threshold_diagnostic": warning_fit, "automatic_deployment": False}
    save_json(args.output/"selection_freeze.json", freeze)
    print(f"[joint train] selected seed={selected['seed']} epoch={selected['best_epoch']}; frozen before test", flush=True)
    reports, prediction_exports = {}, {}

    def evaluate_split(name, dataset, prediction, known_q=None):
        first_steps = collect_failure_steps(dataset, prediction)
        probabilities = known_q if known_q is not None else {
            mode: profile_probabilities(prediction, profile, device) for mode, profile in profiles.items()}
        result_names = {"raw": "raw", "temperature": "temperature_only", "temperature_bias": "temperature_bias_diagnostic"}
        for mode, q in probabilities.items():
            result = report_joint(q, prediction, args.max_examples)
            result.update(probability_mode=mode, temperature=profiles[mode]["temperature"],
                class_biases=profiles[mode]["biases"],
                failure_timing=report_failure_timing(q, prediction, first_steps),
                threshold_warning_diagnostic=threshold_warning_report(q, prediction["labels"],
                    profiles[mode]["warning_threshold_diagnostic"]["threshold"]))
            reports[f"{name}_{result_names[mode]}"] = result
            if mode == args.decision_probability_mode:
                reports[f"{name}_primary"] = result
        # Compatibility name for readers of the previous local V2 report.
        reports[f"{name}_calibrated"] = reports[f"{name}_temperature_bias_diagnostic"]
        if name in ("train", "validation"):
            objective, parts = validation_objective(prediction, weights, args.marginal_weight)
            reports[f"{name}_raw"].update(objective=objective, objective_components=parts,
                objective_scope="selected_fixed_model_eval_mode", ce_reduction=LOSS_REDUCTION)
        if name in args.prediction_splits:
            print(f"[joint train] exporting full {name} predictions and audit times", flush=True)
            prediction_exports[name] = save_full_predictions(args.output/f"{name}_predictions.npz",
                prediction, probabilities, first_steps, dataset, {
                    "split": name, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
                    "architecture": architecture, "selected_training": selected,
                    "decision_probability_mode": args.decision_probability_mode,
                    "profiles": profiles, "source_roots": dataset.source_roots,
                    "source_counts": dataset.source_counts, "test_used_for_selection": False})

    evaluate_split("validation", validation, raw_validation, validation_q)
    del raw_validation, validation_q
    train_loader = make_loader(train, args, device)
    selected_train = collect_logits(model, normalizer, train_loader, device, "selected train diagnostics")
    evaluate_split("train", train, selected_train)
    del selected_train, train_loader
    sets = {"test": test}
    if original["root"] != indexes[0]["root"]:
        sets["original_test"] = JointShardDataset([original], "test", frames)
    for name, dataset in sets.items():
        loader = make_loader(dataset, args, device)
        prediction = collect_logits(model, normalizer, loader, device, name)
        evaluate_split(name, dataset, prediction)
        del prediction, loader
    if file_sha256(checkpoint) != digest:
        raise RuntimeError("checkpoint changed during test reporting")
    assert_inputs_unchanged([original, *indexes], audit)
    train.assert_random_access_unchanged()
    code_paths = [Path(__file__), *(ROOT/"legged_lab/selector"/name for name in
                  ("joint_model.py", "joint_data.py", "joint_training.py", "joint_timing.py", "outcomes.py", "metrics.py"))]
    summary = {"report_version": 4, **freeze, "all_candidates": candidates, "joint_config": asdict(cfg),
        "training_implementation": implementation, "loss_config": loss_config,
        "sampler_config": sampler_config, "training_cache": cache, "training_sources": audit,
        "training_statistics": statistics, "calibration": calibration, "calibration_candidates": profiles,
        "results": reports, "prediction_exports": prediction_exports,
        "validation_predictions": prediction_exports.get("validation"), "train_predictions": prediction_exports.get("train"),
        "inputs_unchanged_after_evaluation": True, "deployment_ready": False,
        "device": str(device), "torch_version": torch.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in code_paths}}
    save_json(args.output/"joint_training_report.json", summary)
    for name, report in reports.items():
        save_json(args.output/"evaluation"/f"{name}.json", report)
    print(f"[joint train] complete: {args.output.resolve()/'joint_training_report.json'}", flush=True)
    print(f"[joint train] V{ARTIFACT_VERSION} checkpoint: {checkpoint.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
