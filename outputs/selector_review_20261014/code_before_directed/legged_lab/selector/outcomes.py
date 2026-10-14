"""Learn the four one-second outcomes without modifying the calibrated risks.

Class order is [both safe, LP-only failure, VP-only failure, both failure].
The second training loss directly compares LP-only and VP-only failure rows.
Neither terrain IDs, fault types nor future measurements are model inputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model import CalibratedSelector, calibrate_probabilities


OUTCOME_ORDER = ("both_safe", "lp_only_failure", "vp_only_failure", "both_failure")


def outcome_classes(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 2 or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("outcome labels must be binary [batch, 2], LP then VP")
    return labels[:, 0].long() + 2 * labels[:, 1].long()


def validate_outcome_probabilities(q: torch.Tensor, rows: int | None = None) -> None:
    if q.ndim != 2 or q.shape[1] != 4 or (rows is not None and len(q) != rows):
        raise ValueError("outcome probabilities must have shape [batch, 4]")
    if (not torch.isfinite(q).all() or not ((q >= 0) & (q <= 1)).all()
            or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-4, rtol=1e-4)):
        raise ValueError("outcome probabilities must be finite and sum to one")


@dataclass(frozen=True)
class OutcomeNetworkConfig:
    frame_hidden_dim: int = 128
    embedding_dim: int = 64
    gru_hidden_dim: int = 64
    head_hidden_dim: int = 96
    dropout: float = 0.1

    def validate(self):
        for name in ("frame_hidden_dim", "embedding_dim", "gru_hidden_dim", "head_hidden_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("outcome dropout must lie in [0, 1)")


class OutcomeNetwork(nn.Module):
    """New trainable temporal encoder plus the frozen risk representation."""

    def __init__(self, frame_dim: int, risk_hidden_dim: int, cfg: OutcomeNetworkConfig | None = None):
        super().__init__()
        self.cfg = cfg or OutcomeNetworkConfig()
        self.cfg.validate()
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_dim, self.cfg.frame_hidden_dim), nn.LayerNorm(self.cfg.frame_hidden_dim), nn.ELU(),
            nn.Linear(self.cfg.frame_hidden_dim, self.cfg.embedding_dim), nn.LayerNorm(self.cfg.embedding_dim), nn.ELU(),
        )
        self.temporal = nn.GRU(self.cfg.embedding_dim, self.cfg.gru_hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.gru_hidden_dim + risk_hidden_dim + 2, self.cfg.head_hidden_dim),
            nn.ELU(), nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.head_hidden_dim, 4),
        )

    def forward(self, history: torch.Tensor, risk_hidden: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
        _, hidden = self.temporal(self.frame_encoder(history))
        risk_logits = torch.logit(risks.float().clamp(1e-6, 1-1e-6))
        return self.head(torch.cat((hidden[-1], risk_hidden, risk_logits), dim=1)).float()


class OutcomeSelector(nn.Module):
    """forward still returns two risks; decision_outputs adds four joint scores."""

    def __init__(self, risk_selector: CalibratedSelector, outcome_model: OutcomeNetwork,
                 temperature: torch.Tensor | None = None, biases: torch.Tensor | None = None):
        super().__init__()
        self.risk_selector = risk_selector
        self.outcome_model = outcome_model
        self.risk_selector.requires_grad_(False).eval()
        temperature = torch.ones(1) if temperature is None else temperature.detach().clone().reshape(-1)
        biases = torch.zeros(4) if biases is None else biases.detach().clone()
        if (temperature.shape != (1,) or not torch.isfinite(temperature).all() or (temperature <= 0).any()
                or biases.shape != (4,) or not torch.isfinite(biases).all()):
            raise ValueError("outcome calibration needs one positive temperature and four finite biases")
        self.register_buffer("outcome_temperature", temperature.float())
        self.register_buffer("outcome_biases", biases.float())

    def train(self, mode: bool = True):
        super().train(mode)
        self.risk_selector.eval()
        return self

    def raw_decision_outputs(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Risk weights and training-only feature statistics remain immutable.
        with torch.no_grad(), torch.autocast(device_type=history.device.type, enabled=False):
            normalized = self.risk_selector.normalizer(history.float())
            risk_hidden = self.risk_selector.model.encode_history(normalized)
            raw_risks = torch.sigmoid(self.risk_selector.model.head(risk_hidden).float())
            risks = calibrate_probabilities(raw_risks, self.risk_selector.temperatures,
                                            self.risk_selector.calibration_biases)
        logits = self.outcome_model(normalized, risk_hidden, risks)
        return risks, logits

    def decision_outputs(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        risks, logits = self.raw_decision_outputs(history)
        return risks, torch.softmax(logits / self.outcome_temperature + self.outcome_biases, dim=1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.risk_selector(history)


def outcome_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor,
                 contrast_weights: torch.Tensor, contrast_coefficient: float = 1.0):
    """Full outcome CE and direction-balanced CE on ONLY contrasting rows."""
    targets = outcome_classes(labels)
    logits = logits.float()
    joint = F.cross_entropy(logits, targets, weight=class_weights)
    contrasting = (targets == 1) | (targets == 2)
    if contrasting.any():
        contrast = F.cross_entropy(logits[contrasting, 1:3], targets[contrasting]-1, weight=contrast_weights)
    else:
        contrast = logits.sum() * 0.
    return joint + contrast_coefficient * contrast, joint, contrast


def weights_from_outcome_counts(counts: torch.Tensor):
    counts = counts.float()
    if counts.shape != (4,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("training needs all four outcomes; class counts must be positive")
    weights = (counts.max()/counts).sqrt().clamp(max=10.)
    contrast = ((counts[1]+counts[2])/(2*counts[1:3])).clamp(max=8.)
    return weights, contrast


def fit_outcome_calibration(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor):
    """Validation-only unweighted multinomial calibration, preserving weights."""
    logits = logits.detach().cpu().double()
    targets = outcome_classes(labels.detach().cpu())
    if logits.shape != (len(targets), 4) or not torch.isfinite(logits).all() or len(targets) == 0:
        raise ValueError("calibration needs finite [samples, 4] logits")
    if (torch.bincount(targets, minlength=4) == 0).any():
        raise ValueError("validation calibration needs all four outcomes")
    w = class_weights.detach().cpu().double()
    if w.shape != (4,) or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("invalid outcome calibration class weights")
    prior_b = -w.log()
    prior_b -= prior_b.mean()
    starts = (torch.zeros(4, dtype=torch.float64), prior_b)
    start = min(starts, key=lambda b: float(F.cross_entropy(logits+b, targets)))
    log_t = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    bias = nn.Parameter(start.clone())
    optimizer = torch.optim.LBFGS([log_t, bias], lr=.5, max_iter=150, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        b = bias.clamp(-30., 30.)
        loss = F.cross_entropy(logits/log_t.exp().clamp(1e-3, 100.) + b-b.mean(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        fitted_b = bias.clamp(-30., 30.)
        fitted_b -= fitted_b.mean()
        candidates = [(torch.ones(1, dtype=torch.float64), b) for b in starts]
        candidates.append((log_t.exp().clamp(1e-3, 100.), fitted_b))
        finite = [(t, b) for t, b in candidates if torch.isfinite(F.cross_entropy(logits/t+b, targets))]
        if not finite:
            raise RuntimeError("outcome calibration failed")
        t, b = min(finite, key=lambda item: float(F.cross_entropy(logits/item[0]+item[1], targets)))
        info = {"method": "multinomial_temperature_and_bias", "fit_split": "validation",
                "uses_test_labels": False, "objective": "unweighted_cross_entropy", "samples": len(targets),
                "before_log_loss": float(F.cross_entropy(logits, targets)),
                "after_log_loss": float(F.cross_entropy(logits/t+b, targets)),
                "temperature": t.float().tolist(), "biases": b.float().tolist()}
    return t.float().clone(), b.float().clone(), info


def outcome_selector_from_artifact(artifact: dict) -> OutcomeSelector:
    from .model import selector_from_artifact
    from .selection import SelectionRule
    if artifact.get("outcome_order") != list(OUTCOME_ORDER):
        raise ValueError("version 5 selector needs the four-outcome contract")
    rule = SelectionRule.from_dict(artifact.get("selection_rule"))
    if rule.mode != "learned_outcomes":
        raise ValueError("version 5 selector requires its learned outcome decision rule")
    risk_artifact = copy.copy(artifact)
    risk_artifact["version"] = 3
    frozen = selector_from_artifact(risk_artifact)
    cfg = OutcomeNetworkConfig(**artifact["outcome_network_config"])
    head = OutcomeNetwork(frozen.model.cfg.frame_dim, frozen.model.cfg.gru_hidden_dim, cfg)
    head.load_state_dict(artifact["outcome_state_dict"], strict=True)
    return OutcomeSelector(frozen, head, artifact["outcome_temperature"], artifact["outcome_biases"]).eval()
