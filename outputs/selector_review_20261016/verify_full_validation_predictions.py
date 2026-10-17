"""Independently verify an exported V6 validation prediction bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(z):
    values = np.exp(z - z.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/home/cjh/桌面"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("full_validation_integrity.json"))
    args = parser.parse_args()
    paths = {name: args.inputs / name for name in (
        "joint_training_report.json", "validation_predictions.json",
        "validation_predictions.npz", "validation_predictions_analysis.json")}
    report = json.loads(paths["joint_training_report.json"].read_text())
    metadata = json.loads(paths["validation_predictions.json"].read_text())
    analysis = json.loads(paths["validation_predictions_analysis.json"].read_text())
    checksums = {name: sha256(path) for name, path in paths.items()}
    assert checksums["validation_predictions.npz"] == metadata["prediction_sha256"] == analysis["prediction_sha256"]
    assert metadata["source"] == analysis["source"]
    assert metadata["source"]["checkpoint_sha256"] == report["checkpoint_sha256_before_test"]
    assert metadata["source"]["selected_training"] == report["selected"]
    assert metadata["source"]["dataset_manifest_sha256"] == report["training_sources"]["base_manifest_sha256"]
    assert metadata["source"]["split"] == "validation"
    assert metadata["primary_probability_mode"] == report["decision_probability_mode"]
    with np.load(paths["validation_predictions.npz"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert set(data) == set(metadata["arrays"])
    for key, array in data.items():
        assert list(array.shape) == metadata["arrays"][key]["shape"], key
        assert str(array.dtype) == metadata["arrays"][key]["dtype"], key
        assert np.isfinite(array).all(), key
    labels, current = data["labels"].astype(bool), data["current_branch"]
    rows = len(labels)
    assert rows == metadata["rows"] == analysis["rows"] == report["training_sources"]["base_counts"]["validation"]
    assert np.isin(data["labels"], [0, 1]).all() and np.isin(current, [0, 1]).all()
    assert np.unique(data["sample_id"]).size == rows
    assert np.array_equal(data["split_row_index"], np.arange(rows))
    shard_ids = data["shard_index"]
    assert np.array_equal(np.unique(shard_ids), np.arange(len(metadata["shard_order"])))
    for index in np.unique(shard_ids):
        positions = np.flatnonzero(shard_ids == index)
        assert np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
        assert np.array_equal(data["row_in_shard"][positions], np.arange(len(positions)))
    target = labels[:, 0].astype(int) + 2 * labels[:, 1].astype(int)
    single = (target == 1) | (target == 2)
    current_failure = np.where(current == 1, labels[:, 0], labels[:, 1])
    need_rescue = single & current_failure
    profiles = {}
    for name, settings in metadata["profiles"].items():
        expected = softmax(data["logits"].astype(float) / np.array(settings["temperature"]) + np.array(settings["biases"]))
        q = data[f"probabilities_{name}"]
        error = float(np.max(np.abs(expected - q)))
        assert error < 1e-6
        assert np.max(np.abs(q.sum(axis=1) - 1)) < 1e-6
        pred = q.argmax(axis=1)
        desired = np.where(pred == 1, 0, np.where(pred == 2, 1, current))
        selected_failure = np.where(desired == 1, labels[:, 0], labels[:, 1])
        switched = desired != current
        calculated = {
            "predicted_class": pred, "desired_branch": desired, "switch_request": switched,
            "warning": pred == 3, "selected_failure": selected_failure,
            "single_failure_wrong": single & selected_failure,
            "missed_rescue": need_rescue & ~switched, "correct_rescue": need_rescue & switched,
            "harmful_switch": single & ~current_failure & switched,
            "wrong_safe_hold": need_rescue & (pred == 0),
            "wrong_both_failure_hold": need_rescue & (pred == 3),
            "wrong_direction": ((target == 1) & (pred == 2)) | ((target == 2) & (pred == 1)),
        }
        for key, value in calculated.items():
            assert np.array_equal(value, data[f"{key}_{name}"]), (name, key)
        confusion = np.bincount(target * 4 + pred, minlength=16).reshape(4, 4)
        assert confusion.tolist() == analysis["profiles"][name]["confusion_rows_true_columns_predicted"]
        for key in ("correct_rescue", "missed_rescue", "harmful_switch", "single_failure_wrong"):
            assert int(calculated[key].sum()) == analysis["profiles"][name][f"{key}_count"]
        profiles[name] = {"max_probability_reconstruction_error": error, "all_row_masks_match": True,
                          "correct_rescue": int(calculated["correct_rescue"].sum()),
                          "missed_rescue": int(calculated["missed_rescue"].sum())}
    result = {"all_checks_passed": True, "rows": rows, "unique_ids": rows,
        "class_counts": np.bincount(target, minlength=4).tolist(), "source_checksums": checksums,
        "checkpoint_sha256": metadata["source"]["checkpoint_sha256"],
        "reported_trained_state_sha256": report["selected"]["trained_state_sha256"],
        "primary_probability_mode": metadata["primary_probability_mode"], "profiles": profiles,
        "scope": "validation predictions only; no model, threshold or test selection"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
