"""Synthetic CPU smoke: collect contract -> three seeds -> evaluate -> export/load."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from legged_lab.selector.config import CollectionConfig
from legged_lab.selector.dataset import (
    SelectorBatch,
    SelectorDatasetWriter,
    split_from_group,
)
from legged_lab.selector.deployment import OnlineSelectorRuntime
from legged_lab.selector.feature_contract import selector_feature_contract
from legged_lab.selector.labels import safety_label_contract
from test_selector_collection_v2 import Adapter


def synthetic_policy_identity():
    return {
        "source_checkpoint_sha256": "a" * 64,
        "reference_policy_sha256": "b" * 64,
        "source_iteration": 60000,
        "actor_observation_dim": 2120,
        "critic_observation_dim": 1037,
        "action_dim": 23,
        "lp_query_history_frames": 5,
        "vp_proprio_history_frames": 10,
        "normalization": "identity",
        "inference_precision": "float32_no_tf32",
    }


def test_cpu_training_evaluation_export_and_artifact_contract(tmp_path):
    root = Path(__file__).parents[2]
    dataset, models, exports = (
        tmp_path / "dataset",
        tmp_path / "models",
        tmp_path / "exported",
    )
    identity = synthetic_policy_identity()
    writer = SelectorDatasetWriter(
        dataset,
        collection_cfg=CollectionConfig(shard_size=16),
        metadata={
            "frozen_policy": identity,
            "safety_label_contract": safety_label_contract(),
            "selector_feature_contract": selector_feature_contract(),
            "collection_complete": True,
        },
    )
    groups = {}
    for index in range(100):
        name = f"synthetic_terrain_{index}"
        groups.setdefault(split_from_group(name), name)
    generator = torch.Generator().manual_seed(42)
    for index, split in enumerate(("train", "validation", "test")):
        features = torch.randn(32, 5, 283, generator=generator)
        labels = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
        ).repeat(8, 1)
        timestamp = torch.ones(32, dtype=torch.float64)
        writer.append(
            SelectorBatch(
                features,
                labels,
                [groups[split]] * 32,
                torch.zeros(32),
                torch.full((32,), index),
                torch.arange(32) + index * 32,
                candidate_time=timestamp,
                feature_end_time=timestamp,
                execution_metadata=torch.zeros(32, 3),
            )
        )
    writer.close()
    env = dict(
        os.environ,
        PYTHONPATH=f"{root}:{root}/rsl_rl",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    def run(script, *args, accepted=(0,)):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "legged_lab/scripts" / script),
                *map(str, args),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode in accepted, result.stdout + result.stderr
        return result

    run(
        "train_selector.py",
        "--dataset",
        dataset,
        "--output",
        models,
        "--device",
        "cpu",
        "--num-workers",
        0,
        "--batch-size",
        16,
        "--max-epochs",
        1,
        "--patience",
        1,
    )
    checkpoint = models / "lp_vp_selector_checkpoint.pt"
    artifact = torch.load(checkpoint, weights_only=False)
    assert (
        artifact["version"] == 2
        and artifact["selector_feature_contract"] == selector_feature_contract()
    )
    assert all((models / f"selector_seed_{seed}.pt").is_file() for seed in (42, 43, 44))
    # Random synthetic labels are not expected to pass the deployment-quality gate.
    run(
        "eval_selector.py",
        "--dataset",
        dataset,
        "--checkpoint",
        checkpoint,
        "--device",
        "cpu",
        "--output",
        tmp_path / "evaluation.json",
        accepted=(0, 2),
    )
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["samples"] == 32 and len(evaluation["checks"]) == 3
    run("export_selector.py", "--checkpoint", checkpoint, "--output-dir", exports)
    extra = {"selector_contract.json": ""}
    model = torch.jit.load(str(exports / "lp_vp_selector.pt"), _extra_files=extra)
    assert (
        json.loads(extra["selector_contract.json"])["selector_feature_contract"]
        == selector_feature_contract()
    )
    for batch in (1, 8):
        risks = model(torch.randn(batch, 5, 283))
        assert risks.shape == (batch, 4) and torch.isfinite(risks).all()
        assert (risks[:, 0] <= risks[:, 1]).all() and (risks[:, 2] <= risks[:, 3]).all()
    dual_path = exports / "renet_dual_branch.pt"
    dual = torch.jit.trace(Adapter(), torch.zeros(2, 2120))
    torch.jit.save(
        dual,
        str(dual_path),
        _extra_files={
            "renet_dual_branch_contract.json": json.dumps({"frozen_policy": identity})
        },
    )
    assert isinstance(
        OnlineSelectorRuntime.from_artifacts(dual_path, exports / "lp_vp_selector.pt"),
        OnlineSelectorRuntime,
    )
