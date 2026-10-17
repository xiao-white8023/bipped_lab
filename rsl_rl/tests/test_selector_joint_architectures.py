"""CPU regressions for feature routing, trainable readouts and V6/V7 artifacts."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from legged_lab.selector.feature_contract import selector_feature_contract
from legged_lab.selector.joint_model import (
    ARCHITECTURES, ARTIFACT_KIND, ARTIFACT_VERSION, JointOutcomeNetwork, JointSelectorConfig,
    changed_components, cpu_state, joint_selector_from_artifact, joint_selector_loss,
    risks_from_joint, split_joint_features,
)
from legged_lab.selector.labels import safety_label_contract
from legged_lab.selector.model import FeatureNormalizer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_feature_split_covers_each_existing_column_once():
    x = torch.arange(283.).expand(2, 5, -1)
    common, lp, vp = split_joint_features(x)
    expected = [list(range(78))+list(range(274, 283)),
        list(range(78, 142))+list(range(206, 209))+list(range(212, 235))+list(range(258, 266)),
        list(range(142, 206))+list(range(209, 212))+list(range(235, 258))+list(range(266, 274))]
    assert (common.shape, lp.shape, vp.shape) == ((2, 5, 87), (2, 5, 98), (2, 5, 98))
    for actual, columns in zip((common, lp, vp), expected):
        torch.testing.assert_close(actual[0, 0], torch.tensor(columns).float())
    assert sorted(sum(expected, [])) == list(range(283))


@pytest.mark.parametrize('architecture', ARCHITECTURES)
@pytest.mark.parametrize('frames', [5, 10, 20])
def test_architecture_shapes_batch_independence_and_marginals(architecture, frames):
    model = JointOutcomeNetwork(JointSelectorConfig(architecture=architecture, history_frames=frames)).eval()
    x = torch.randn(3, frames, 283)
    with torch.no_grad():
        logits = model(x)
        torch.testing.assert_close(logits[:1], model(x[:1]), rtol=1e-5, atol=1e-6)
    assert logits.shape == (3, 4) and logits.dtype == torch.float32
    q = logits.softmax(1)
    risks = risks_from_joint(q)
    torch.testing.assert_close(risks[:, 0], q[:, 1]+q[:, 3])
    torch.testing.assert_close(risks[:, 1], q[:, 2]+q[:, 3])


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_all_components_and_history_receive_gradients(architecture):
    torch.manual_seed(17)
    model = JointOutcomeNetwork(JointSelectorConfig(architecture=architecture))
    initial = cpu_state(model)
    x = torch.randn(8, 5, 283, requires_grad=True)
    labels = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).repeat(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = joint_selector_loss(model(x), labels, [0.7, 2.3, 3.1, 3.4])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum(dim=(0, 2)) > 0).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    proof = changed_components(initial, cpu_state(model), model.cfg.gru_layers)
    assert all(count > 0 for count in proof.values())
    if architecture == 'branch_gru':
        for prefix in ['common_encoder', 'lp_encoder', 'vp_encoder', 'lp_attention', 'vp_attention',
                       'lp_gru_layer_0', 'vp_gru_layer_0', 'head']:
            assert prefix in proof
        for branch in ['lp_attention', 'vp_attention']:
            weights = getattr(model, branch).weights(torch.randn(8, 5, 128))
            torch.testing.assert_close(weights.sum(1), torch.ones(8, 1))


class _OriginalV6Network(nn.Module):
    """The pre-change V6 topology, independent of the new architecture factory."""
    def __init__(self):
        super().__init__()
        self.frame_encoder = nn.Sequential(nn.Linear(283, 256), nn.LayerNorm(256), nn.ELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ELU())
        self.temporal = nn.GRU(128, 128, num_layers=3, batch_first=True, dropout=.1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4))

    def forward(self, x):
        _, hidden = self.temporal(self.frame_encoder(x))
        return self.head(hidden[-1]).float()


def _artifact(model, version=ARTIFACT_VERSION):
    normalizer = FeatureNormalizer()
    normalizer.set_statistics(torch.randn(283), torch.rand(283)+.5)
    cfg = asdict(model.cfg)
    if version == 6:
        cfg = {k: cfg[k] for k in ['history_frames', 'frame_dim', 'frame_hidden_dims',
            'gru_hidden_dim', 'gru_layers', 'gru_dropout', 'head_hidden_dim']}
    return {'version': version, 'artifact_kind': ARTIFACT_KIND if version == ARTIFACT_VERSION else 'single_encoder_joint_outcomes',
        'joint_config': cfg, 'outcome_order': ['both_safe', 'lp_only_failure', 'vp_only_failure', 'both_failure'],
        'model_state_dict': cpu_state(model), 'normalizer_state_dict': cpu_state(normalizer),
        'joint_temperature': torch.tensor([1.2]), 'joint_biases': torch.tensor([.2, -.1, .3, -.4])}


def test_v6_initialization_and_saved_predictions_are_preserved():
    torch.manual_seed(91)
    old = _OriginalV6Network().eval()
    torch.manual_seed(91)
    new = JointOutcomeNetwork().eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    artifact = _artifact(new, version=6)
    restored = joint_selector_from_artifact(artifact)
    x = torch.randn(7, 5, 283)
    with torch.no_grad():
        expected = torch.softmax(old(restored.normalizer(x))/1.2+artifact['joint_biases'], 1)
        _, actual = restored.decision_outputs(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_v7_roundtrip_and_architecture_identity(tmp_path, architecture):
    model = JointOutcomeNetwork(JointSelectorConfig(architecture=architecture)).eval()
    artifact = _artifact(model)
    path = tmp_path/'selector.pt'
    torch.save(artifact, path)
    restored = joint_selector_from_artifact(torch.load(path, weights_only=False)).eval()
    x = torch.randn(3, 5, 283)
    with torch.no_grad():
        torch.testing.assert_close(restored.raw_logits(x), model(restored.normalizer(x)), rtol=0, atol=0)
    assert restored.model.cfg.architecture == architecture
    artifact['version'] = 6
    artifact['artifact_kind'] = 'single_encoder_joint_outcomes'
    if architecture != 'single_gru':
        with pytest.raises(ValueError, match='V6'):
            joint_selector_from_artifact(artifact)


def _make_dataset(root):
    identity = {'source_checkpoint_sha256': 'a'*64, 'reference_policy_sha256': 'b'*64,
        'source_iteration': 60000, 'actor_observation_dim': 2120, 'critic_observation_dim': 1037,
        'action_dim': 23, 'lp_query_history_frames': 5, 'vp_proprio_history_frames': 10,
        'normalization': 'identity', 'inference_precision': 'float32_no_tf32'}
    manifest = {'version': 4, 'feature_shape': [5, 283], 'label_order': ['lp_1.0s', 'vp_1.0s'],
        'metadata': {'collection_complete': True, 'frozen_policy': identity,
            'selector_feature_contract': selector_feature_contract(), 'safety_label_contract': safety_label_contract()},
        'counts': {}, 'shards': {}, 'group_sha256': {}}
    for split_index, split in enumerate(['train', 'validation', 'test']):
        n = 32 if split == 'train' else 16
        directory = root/split
        directory.mkdir(parents=True)
        y = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).repeat(n//4, 1)
        x = torch.randn(n, 5, 283, generator=torch.Generator().manual_seed(11+split_index))
        x[:, :, -2] = (torch.arange(n)//4 % 2).float()[:, None]
        future = torch.zeros(n, 2, 10)
        steps = torch.tensor([10, 25, 35, 50]).repeat_interleave(4).repeat(n//16)[:, None].expand(-1, 2)
        future[:, :, 0] = torch.where(y.bool(), steps, -1)
        group = f'terrain_seed5:type{split_index}:level0'
        torch.save({'features': x, 'labels': y, 'scenario_seed': torch.arange(n)+split_index*1000,
            'group_ids': [group]*n, 'future_metrics': future}, directory/'shard_000000.pt')
        manifest['counts'][split], manifest['shards'][split] = n, 1
        manifest['group_sha256'][split] = hashlib.sha256(group.encode()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('probability_mode', ['raw', 'temperature', 'temperature_bias'])
def test_train_save_evaluate_and_validation_export_end_to_end(tmp_path, probability_mode):
    from legged_lab.scripts.train_selector_joint import main as train_main
    from legged_lab.scripts.eval_selector_joint import main as eval_main
    from legged_lab.scripts.export_selector_joint_validation_predictions import main as export_main

    dataset, output = tmp_path/'dataset', tmp_path/'run'
    _make_dataset(dataset)
    arguments = ['--dataset', str(dataset), '--output', str(output), '--device', 'cpu',
        '--seeds', '42', '--max-epochs', '1', '--patience', '1', '--batch-size', '8',
        '--num-workers', '0', '--max-examples', '2']
    if probability_mode != 'raw':
        arguments.extend(['--decision-probability-mode', probability_mode])
    assert train_main(arguments) == 0
    report = json.loads((output/'joint_training_report.json').read_text())
    assert report['architecture']['name'] == 'branch_gru'
    assert report['selected']['all_network_parameters_trainable']
    assert report['decision_probability_mode'] == probability_mode
    assert report['calibration_fit_split'] == (None if probability_mode == 'raw' else 'validation')
    assert report['test_used_for_selection'] is False
    checkpoint = output/'joint_selector_checkpoint.pt'
    artifact = torch.load(checkpoint, weights_only=False)
    assert artifact['version'] == 7
    assert artifact['joint_temperature'].tolist() == report['calibration_candidates'][probability_mode]['temperature']
    assert artifact['joint_biases'].tolist() == report['calibration_candidates'][probability_mode]['biases']
    cache = json.loads((output/'training_cache.json').read_text())
    assert set(cache['arrays']) == {'features', 'labels', 'sample_id'}
    for split, n in [('train', 32), ('validation', 16)]:
        prediction = np.load(output/f'{split}_predictions.npz', allow_pickle=False)
        assert prediction['logits'].shape == (n, 4) and prediction['first_failure_steps'].shape == (n, 2)
        np.testing.assert_array_equal(prediction['probabilities_raw'].argmax(1), prediction['predicted_class_raw'])
        assert report['results'][split+'_primary']['probability_mode'] == probability_mode
        rows = report['results'][split+'_raw']['failure_timing']['rows']
        assert len(rows) == 15 and all('predicted_class_counts' in row for row in rows)
        for row in rows:
            assert sum(row['predicted_class_counts']) == row['samples']
    evaluated = tmp_path/'evaluation.json'
    assert eval_main(['--dataset', str(dataset), '--checkpoint', str(checkpoint), '--device', 'cpu',
        '--output', str(evaluated)]) == 0
    evaluation = json.loads(evaluated.read_text())
    assert evaluation['confusion_rows_true_columns_predicted'] == report['results']['test_primary']['confusion_rows_true_columns_predicted']
    exported = tmp_path/'exported_validation.pt'
    assert export_main(['--dataset', str(dataset), '--checkpoint', str(checkpoint), '--output', str(exported), '--device', 'cpu']) == 0
    dump = torch.load(exported, weights_only=False)
    expected = torch.softmax(dump['raw_logits']/artifact['joint_temperature']+artifact['joint_biases'], 1)
    torch.testing.assert_close(expected, dump['calibrated_q'])


@pytest.mark.parametrize('architecture,layers', [('branch_gru', '1'), ('single_gru', '3'), ('single_gru_readout', '3')])
def test_launcher_passes_structure_and_raw_mode(tmp_path, architecture, layers):
    stub = tmp_path/'python_stub'
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('SELECTOR_')}
    environment.update(SELECTOR_PYTHON=str(stub), SELECTOR_ARCHITECTURE=architecture,
        SELECTOR_JOINT_OUTPUT=str(tmp_path/'new_run'), SELECTOR_USE_SUPPLEMENTS='0',
        SELECTOR_VERIFY_FILTERED_LINEAGE='0', SELECTOR_SEEDS='7 8')
    script = Path(__file__).resolve().parents[2]/'legged_lab/selector/run_selector_joint_training.sh'
    result = subprocess.run(['bash', str(script)], env=environment, text=True, capture_output=True, check=True)
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index('--architecture')+1] == architecture
    assert arguments[arguments.index('--gru-layers')+1] == layers
    assert arguments[arguments.index('--decision-probability-mode')+1] == 'raw'
    seed_index = arguments.index('--seeds')
    assert arguments[seed_index+1:seed_index+3] == ['7', '8']


def test_invalid_architecture_and_history_are_rejected():
    with pytest.raises(ValueError, match='architecture'):
        JointOutcomeNetwork(JointSelectorConfig(architecture='unknown'))
    model = JointOutcomeNetwork(JointSelectorConfig(architecture='branch_gru'))
    with pytest.raises(ValueError, match='expected'):
        model(torch.randn(2, 10, 283))
