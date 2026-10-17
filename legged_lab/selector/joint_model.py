"""Temporal selector architectures with one joint distribution and two risks.

No risk-selector checkpoint is loaded. The four logits supervise the complete
encoders/GRUs/readouts/head, including optional marginal LP/VP risk losses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model import FeatureNormalizer
from .outcomes import OUTCOME_ORDER, outcome_classes, validate_outcome_probabilities


ARTIFACT_VERSION = 7
ARTIFACT_KIND = "temporal_joint_outcomes"
LEGACY_ARTIFACT_KIND = "single_encoder_joint_outcomes"
ARCHITECTURES = ("single_gru", "single_gru_readout", "branch_gru")


@dataclass(frozen=True)
class JointSelectorConfig:
    history_frames: int = 5
    frame_dim: int = 283
    frame_hidden_dims: tuple[int, int] = (256, 128)
    gru_hidden_dim: int = 128
    gru_layers: int | None = None
    gru_dropout: float = .1
    head_hidden_dim: int = 64
    # Missing fields in an existing V6 configuration preserve its exact model.
    architecture: str = "single_gru"
    common_hidden_dim: int = 128
    common_embedding_dim: int = 64
    branch_embedding_dim: int = 128
    attention_hidden_dim: int = 64
    fusion_hidden_dim: int = 256

    def __post_init__(self):
        if self.gru_layers is None:
            object.__setattr__(self, "gru_layers", 1 if self.architecture == "branch_gru" else 3)

    def validate(self):
        if self.history_frames not in (5, 10, 20) or self.frame_dim != 283:
            raise ValueError("joint selector requires 5/10/20 actual frames of 283 features")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown joint selector architecture: {self.architecture}")
        dimensions = (*self.frame_hidden_dims, self.gru_hidden_dim, self.gru_layers, self.head_hidden_dim,
                      self.common_hidden_dim, self.common_embedding_dim, self.branch_embedding_dim,
                      self.attention_hidden_dim, self.fusion_hidden_dim)
        if len(self.frame_hidden_dims) != 2 or any(type(n) is not int or n < 1 for n in dimensions):
            raise ValueError("encoder/GRU/head dimensions and layer count must be positive integers")
        if not math.isfinite(self.gru_dropout) or not 0 <= self.gru_dropout < 1:
            raise ValueError("GRU dropout must lie in [0, 1)")


def split_joint_features(history):
    """Split the existing observable contract; no audit/future fields enter here.

    Public: proprio78 + cross7 + state2 = 87.
    Each branch: latent64 + velocity3 + action23 + health8 = 98.
    Input may already be normalized: the original column positions are retained.
    """
    if history.ndim != 3 or history.shape[-1] != 283:
        raise ValueError("joint features must have shape [batch,time,283]")
    common = torch.cat((history[..., :78], history[..., 274:283]), dim=-1)
    lp = torch.cat((history[..., 78:142], history[..., 206:209],
                    history[..., 212:235], history[..., 258:266]), dim=-1)
    vp = torch.cat((history[..., 142:206], history[..., 209:212],
                    history[..., 235:258], history[..., 266:274]), dim=-1)
    return common, lp, vp


def _frame_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim), nn.ELU(),
    )


class TemporalAttentionPool(nn.Module):
    """A trainable weighted sum over the provided past/current GRU outputs."""
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh(),
                                   nn.Linear(attention_dim, 1, bias=False))

    def weights(self, sequence):
        # Stable probabilities under autocast; these are readout weights, not risks.
        return torch.softmax(self.score(sequence).float(), dim=1)

    def forward(self, sequence):
        return (sequence * self.weights(sequence).to(sequence.dtype)).sum(dim=1)


class JointOutcomeNetwork(nn.Module):
    """Forward returns raw [B,4] logits, not class probabilities."""
    def __init__(self, cfg: JointSelectorConfig | None = None):
        super().__init__()
        self.cfg = cfg or JointSelectorConfig()
        self.cfg.validate()
        cfg = self.cfg
        first, second = cfg.frame_hidden_dims
        dropout = cfg.gru_dropout if cfg.gru_layers > 1 else 0.
        if cfg.architecture == "branch_gru":
            self.common_encoder = _frame_mlp(87, cfg.common_hidden_dim, cfg.common_embedding_dim)
            branch_input = 98+cfg.common_embedding_dim
            self.lp_encoder = _frame_mlp(branch_input, first, cfg.branch_embedding_dim)
            self.vp_encoder = _frame_mlp(branch_input, first, cfg.branch_embedding_dim)
            self.lp_temporal = nn.GRU(cfg.branch_embedding_dim, cfg.gru_hidden_dim,
                num_layers=cfg.gru_layers, batch_first=True, dropout=dropout)
            self.vp_temporal = nn.GRU(cfg.branch_embedding_dim, cfg.gru_hidden_dim,
                num_layers=cfg.gru_layers, batch_first=True, dropout=dropout)
            self.lp_attention = TemporalAttentionPool(cfg.gru_hidden_dim, cfg.attention_hidden_dim)
            self.vp_attention = TemporalAttentionPool(cfg.gru_hidden_dim, cfg.attention_hidden_dim)
            readout_dim = 2*(2*cfg.gru_hidden_dim+cfg.branch_embedding_dim)+cfg.common_embedding_dim
            self.head = nn.Sequential(
                nn.Linear(readout_dim, cfg.fusion_hidden_dim), nn.LayerNorm(cfg.fusion_hidden_dim), nn.ELU(),
                nn.Linear(cfg.fusion_hidden_dim, cfg.head_hidden_dim), nn.ELU(), nn.Linear(cfg.head_hidden_dim, 4))
        else:
            # Preserve V6 parameter names, shapes and initialization order exactly.
            self.frame_encoder = _frame_mlp(cfg.frame_dim, first, second)
            self.temporal = nn.GRU(second, cfg.gru_hidden_dim, num_layers=cfg.gru_layers,
                batch_first=True, dropout=dropout)
            readout_dim = cfg.gru_hidden_dim
            if cfg.architecture == "single_gru_readout":
                self.attention_pool = TemporalAttentionPool(cfg.gru_hidden_dim, cfg.attention_hidden_dim)
                readout_dim = 2*cfg.gru_hidden_dim+second
            self.head = nn.Sequential(nn.Linear(readout_dim, cfg.head_hidden_dim),
                                      nn.ELU(), nn.Linear(cfg.head_hidden_dim, 4))

    def forward(self, history):
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.cfg.history_frames, self.cfg.frame_dim):
            raise ValueError(f"expected [batch,{self.cfg.history_frames},{self.cfg.frame_dim}] real history")
        if self.cfg.architecture == "branch_gru":
            public, lp, vp = split_joint_features(history)
            common = self.common_encoder(public)
            lp_frames = self.lp_encoder(torch.cat((common, lp), dim=-1))
            vp_frames = self.vp_encoder(torch.cat((common, vp), dim=-1))
            lp_sequence, lp_hidden = self.lp_temporal(lp_frames)
            vp_sequence, vp_hidden = self.vp_temporal(vp_frames)
            summary = torch.cat((lp_hidden[-1], self.lp_attention(lp_sequence), lp_frames[:, -1],
                                 vp_hidden[-1], self.vp_attention(vp_sequence), vp_frames[:, -1],
                                 common[:, -1]), dim=-1)
        else:
            frames = self.frame_encoder(history)
            sequence, hidden = self.temporal(frames)
            summary = hidden[-1]
            if self.cfg.architecture == "single_gru_readout":
                summary = torch.cat((summary, self.attention_pool(sequence), frames[:, -1]), dim=-1)
        return self.head(summary).float()

    def architecture_description(self):
        cfg = self.cfg
        return {"name": cfg.architecture, "history_frames": cfg.history_frames,
            "explicit_history_span_s": (cfg.history_frames-1)*.1,
            "parameters": sum(p.numel() for p in self.parameters()),
            "gru_layers_per_stream": cfg.gru_layers,
            "temporal_streams": 2 if cfg.architecture == "branch_gru" else 1,
            "readout": "last_state" if cfg.architecture == "single_gru" else
                       "last_state_plus_attention_over_all_frames_plus_current_frame",
            "common_input_dim": 87 if cfg.architecture == "branch_gru" else None,
            "branch_input_dim": 98 if cfg.architecture == "branch_gru" else None,
            "output": "four_joint_logits_with_exact_LP_VP_marginals"}


def marginal_logits(logits):
    """Stable logit(q1+q3), logit(q2+q3), without probability clipping."""
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("joint logits must have shape [batch,4]")
    z = logits.float()
    lp = torch.logsumexp(z[:, [1, 3]], 1) - torch.logsumexp(z[:, [0, 2]], 1)
    vp = torch.logsumexp(z[:, [2, 3]], 1) - torch.logsumexp(z[:, [0, 1]], 1)
    return torch.stack((lp, vp), dim=1)


def joint_selector_loss(logits, labels, class_weights=None, marginal_weight=1.):
    """CE + lambda*(LP BCE+VP BCE)/2, through the SAME four logits.

    CE is mean_i(weight[target_i] * NLL_i), NOT sum(weight*NLL)/sum(batch weights).
    Training supplies weights normalized once to expectation one on train rows;
    the exact same frozen weights and row-mean reduction apply to validation.
    No direction-only loss, independent risk head, or switching penalty.
    Returned components retain gradients for callers wanting diagnostics.
    """
    if not math.isfinite(marginal_weight) or marginal_weight < 0:
        raise ValueError("marginal weight must be finite and nonnegative")
    if logits.ndim != 2 or logits.shape != (len(labels), 4) or len(labels) == 0:
        raise ValueError("need nonempty matching [B,4] logits and [B,2] labels")
    z = logits.float()
    y = labels.to(device=z.device, dtype=torch.float32)
    target = outcome_classes(y)
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=z.dtype, device=z.device)
    if w is not None and (w.shape != (4,) or not torch.isfinite(w).all() or (w <= 0).any()):
        raise ValueError("four positive finite training class weights required")
    ce = F.cross_entropy(z, target, weight=w, reduction="none").mean()
    bce = F.binary_cross_entropy_with_logits(marginal_logits(z), y, reduction="none").mean(0)
    return ce + marginal_weight * bce.mean(), {"joint_ce": ce, "lp_bce": bce[0], "vp_bce": bce[1]}


def risks_from_joint(q):
    validate_outcome_probabilities(q)
    return torch.stack((q[:, 1] + q[:, 3], q[:, 2] + q[:, 3]), dim=1)


def propose_joint_argmax(q, current_mask):
    """Safe->hold, LP-only->VP, VP-only->LP, both->hold+warn."""
    validate_outcome_probabilities(q)
    current = current_mask.to(q.device).reshape(-1)
    if len(current) != len(q) or not ((current == 0) | (current == 1)).all():
        raise ValueError("current masks must be VP=0/LP=1")
    target = q.argmax(1)
    desired = current.clone()
    desired[target == 1] = 0
    desired[target == 2] = 1
    return desired, target == 3


class CalibratedJointSelector(nn.Module):
    """Joint calibration first, marginalization second: consistency is exact."""
    def __init__(self, model, normalizer=None, temperature=None, biases=None):
        super().__init__()
        self.model = model
        self.normalizer = normalizer if normalizer is not None else FeatureNormalizer(model.cfg.frame_dim)
        t = torch.ones(1) if temperature is None else torch.as_tensor(temperature).detach().clone().reshape(-1)
        b = torch.zeros(4) if biases is None else torch.as_tensor(biases).detach().clone()
        if (t.shape != (1,) or not torch.isfinite(t).all() or (t <= 0).any()
                or b.shape != (4,) or not torch.isfinite(b).all()):
            raise ValueError("joint calibration needs one positive temperature and four finite biases")
        self.register_buffer("temperature", t.float())
        self.register_buffer("biases", b.float())

    def raw_logits(self, history):
        return self.model(self.normalizer(history.float()))

    def decision_outputs(self, history):
        z = self.raw_logits(history)
        q = torch.softmax(z / self.temperature + self.biases, 1)
        return risks_from_joint(q), q

    def forward(self, history):
        risks, _ = self.decision_outputs(history)
        return risks


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def changed_components(initial, trained, layers):
    if initial.keys() != trained.keys():
        raise RuntimeError("initial and trained network state keys differ")
    groups = {}
    for prefix in ("frame_encoder", "common_encoder", "lp_encoder", "vp_encoder",
                   "attention_pool", "lp_attention", "vp_attention", "head"):
        keys = [k for k in initial if k.startswith(prefix+".")]
        if keys:
            groups[prefix] = keys
    for prefix in ("temporal", "lp_temporal", "vp_temporal"):
        if any(k.startswith(prefix+".") for k in initial):
            for layer in range(layers):
                name = "gru" if prefix == "temporal" else prefix.replace("temporal", "gru")
                groups[f"{name}_layer_{layer}"] = [k for k in initial if k.startswith(prefix+".") and k.endswith(f"_l{layer}")]
    covered = {k for keys in groups.values() for k in keys}
    if covered != set(initial) or any(not keys for keys in groups.values()):
        raise RuntimeError("network update audit does not cover every component")
    changed = {name: sum(not torch.equal(initial[k], trained[k]) for k in keys) for name, keys in groups.items()}
    if any(not value for value in changed.values()):
        raise RuntimeError(f"some trainable component did not update: {changed}")
    return changed


def joint_selector_from_artifact(artifact):
    identity = (artifact.get("version"), artifact.get("artifact_kind"))
    if identity not in ((ARTIFACT_VERSION, ARTIFACT_KIND), (6, LEGACY_ARTIFACT_KIND)):
        raise ValueError("expected a V7 temporal joint selector or a V6 single-encoder joint selector")
    if artifact.get("outcome_order") != list(OUTCOME_ORDER):
        raise ValueError("joint outcome order differs")
    cfg = JointSelectorConfig(**artifact["joint_config"])
    cfg.validate()
    if identity[0] == 6 and cfg.architecture != "single_gru":
        raise ValueError("V6 checkpoints require the original single_gru architecture")
    model = JointOutcomeNetwork(cfg)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    normalizer = FeatureNormalizer(cfg.frame_dim)
    normalizer.load_state_dict(artifact["normalizer_state_dict"], strict=True)
    if (not bool(normalizer.fitted) or not torch.isfinite(normalizer.mean).all()
            or not torch.isfinite(normalizer.std).all() or (normalizer.std <= 0).any()):
        raise ValueError("invalid or unfitted training-only feature statistics")
    return CalibratedJointSelector(model, normalizer, artifact["joint_temperature"], artifact["joint_biases"]).eval()
