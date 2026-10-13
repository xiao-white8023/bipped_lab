"""Small calibrated recurrent risk model used by the LP/VP selector."""

from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .config import SelectorConfig

# 保存仅由训练集拟合的逐特征均值和标准差，并执行归一化
class FeatureNormalizer(nn.Module):
    """Serializable per-feature normalization fitted on the training split."""

    def __init__(self, frame_dim: int = 283, epsilon: float = 1.0e-6):
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(self.frame_dim))
        self.register_buffer("std", torch.ones(self.frame_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        if features.shape[-1] != self.frame_dim:
            raise ValueError(f"features must end in {self.frame_dim} dimensions")
        flattened = features.reshape(-1, self.frame_dim).float()
        finite = torch.isfinite(flattened)
        safe = torch.where(finite, flattened, torch.zeros_like(flattened))
        count = finite.sum(dim=0).clamp_min(1)
        mean = safe.sum(dim=0) / count
        variance = (torch.where(finite, flattened - mean, torch.zeros_like(flattened)).square()).sum(dim=0) / count
        self.mean.copy_(mean)
        self.std.copy_(variance.sqrt().clamp_min(self.epsilon))
        self.fitted.fill_(True)

    @torch.no_grad()
    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError("normalizer statistics have the wrong shape")
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.epsilon))
        self.fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.frame_dim:
            raise ValueError(f"features must end in {self.frame_dim} dimensions")
        safe = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return (safe - self.mean) / self.std

# 建立均值、标准差和已拟合标记等可随模型保存的缓冲。
class SelectorNetwork(nn.Module):
    """Frame MLP + GRU predicting [LP_1s, VP_1s] failure probabilities."""

    def __init__(self, cfg: SelectorConfig | None = None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg or SelectorConfig())
        self.cfg.validate()
        first, second = self.cfg.frame_hidden_dims
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.cfg.frame_dim, first),
            nn.LayerNorm(first),
            nn.ELU(),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.ELU(),
        )
        self.temporal = nn.GRU(
            input_size=second,
            hidden_size=self.cfg.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.cfg.gru_hidden_dim, self.cfg.head_hidden_dim),
            nn.ELU(),
            nn.Linear(self.cfg.head_hidden_dim, self.cfg.output_dim),
        )

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("selector history must have shape [batch, time, frame_dim]")
        if history.shape[1] != self.cfg.history_frames or history.shape[2] != self.cfg.frame_dim:
            raise ValueError(
                f"selector history must have shape [batch, {self.cfg.history_frames}, {self.cfg.frame_dim}]"
            )
        encoded = self.frame_encoder(history)
        _, hidden = self.temporal(encoded)
        return hidden[-1]

    def raw_outputs(self, history: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_history(history))

    @staticmethod
    def probabilities_from_raw(raw: torch.Tensor) -> torch.Tensor:
        """Convert the independent LP/VP logits to one-second risks."""

        # Keep probability arithmetic in FP32 under mixed precision.
        return torch.sigmoid(raw.float())

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.probabilities_from_raw(self.raw_outputs(history))


class CalibratedSelector(nn.Module):
    """Exportable normalizer and two affine logit calibrators.

    Missing biases mean zero, preserving old temperature-only checkpoints.
    """

    def __init__(
        self,
        model: SelectorNetwork,
        normalizer: FeatureNormalizer | None = None,
        temperatures: torch.Tensor | None = None,
        calibration_biases: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer or FeatureNormalizer(model.cfg.frame_dim)
        if temperatures is None:
            temperatures = torch.ones(2)
        if temperatures.shape != (2,):
            raise ValueError("temperatures must have shape [2]")
        if not torch.isfinite(temperatures).all() or (temperatures <= 0).any():
            raise ValueError("temperatures must be finite and positive")
        if calibration_biases is None:
            calibration_biases = torch.zeros_like(temperatures)
        if calibration_biases.shape != (2,) or not torch.isfinite(calibration_biases).all():
            raise ValueError("calibration_biases must have shape [2] and be finite")
        self.register_buffer("temperatures", temperatures.float().clamp_min(1.0e-3))
        self.register_buffer("calibration_biases", calibration_biases.float())

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        probabilities = self.model(self.normalizer(history))
        return calibrate_probabilities(probabilities, self.temperatures, self.calibration_biases)


def calibrate_probabilities(
    probabilities: torch.Tensor, temperatures: torch.Tensor, biases: torch.Tensor,
) -> torch.Tensor:
    logits = torch.logit(probabilities.float().clamp(1.0e-6, 1.0 - 1.0e-6))
    return torch.sigmoid(logits / temperatures + biases)


def selector_from_artifact(artifact: dict) -> CalibratedSelector:
    """Load both legacy two-head checkpoints and affine-calibrated ones."""
    from .feature_contract import validate_feature_contract
    from .labels import validate_safety_label_contract

    validate_feature_contract(artifact.get("selector_feature_contract"))
    validate_safety_label_contract(artifact.get("safety_label_contract"))
    if artifact.get("version") == 5:
        from .outcomes import outcome_selector_from_artifact
        return outcome_selector_from_artifact(artifact)
    if artifact.get("version") not in (3, 4):
        raise ValueError("selector requires a two-output checkpoint, version 3 or 4")
    if artifact.get("version") == 4:
        if "calibration_biases" not in artifact or "selection_rule" not in artifact:
            raise ValueError("version 4 selector needs calibration_biases and selection_rule")
        from .selection import SelectionRule
        if SelectionRule.from_dict(artifact["selection_rule"]).mode == "learned_outcomes":
            raise ValueError("learned outcomes require checkpoint version 5")
    cfg = SelectorConfig(**artifact["selector_config"])
    model = SelectorNetwork(cfg)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    normalizer = FeatureNormalizer(cfg.frame_dim)
    normalizer.load_state_dict(artifact["normalizer_state_dict"], strict=True)
    return CalibratedSelector(model, normalizer, artifact["temperatures"],
                              artifact.get("calibration_biases")).eval()



class SelectorEnsemble(nn.Module):
    """Three independently seeded calibrated models for mean risk and uncertainty."""

    def __init__(self, models: list[CalibratedSelector]):
        super().__init__()
        if not models:
            raise ValueError("selector ensemble cannot be empty")
        self.models = nn.ModuleList(models)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.stack([model(history) for model in self.models], dim=0)
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)


def weighted_binary_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape != targets.shape or probabilities.shape[-1] != 2:
        raise ValueError("probabilities and targets must have matching [batch, 2] shapes")
    if positive_weights.shape != (2,):
        raise ValueError("positive_weights must have shape [2]")
    probabilities = probabilities.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    targets = targets.to(probabilities.dtype)
    loss = -(
        positive_weights[None] * targets * torch.log(probabilities)
        + (1.0 - targets) * torch.log1p(-probabilities)
    )
    return loss.mean()


@torch.no_grad()
def positive_class_weights(labels: torch.Tensor, maximum: float = 100.0) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("labels must have shape [samples, 2]")
    positive = labels.float().sum(dim=0)
    negative = labels.shape[0] - positive
    return (negative / positive.clamp_min(1.0)).clamp(1.0, maximum)


def fit_temperatures(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Fit two positive scalar temperatures on a validation split."""

    if probabilities.shape != labels.shape or probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("validation probabilities and labels must have shape [samples, 2]")
    logits = torch.logit(probabilities.detach().clamp(1.0e-6, 1.0 - 1.0e-6))
    log_temperature = nn.Parameter(torch.zeros(2, device=probabilities.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1.0e-3, 100.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(1.0e-3, 100.0)


def fit_affine_calibration(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_weights: torch.Tensor | None = None,
    initial_temperatures: torch.Tensor | None = None,
    initial_biases: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit positive temperatures AND offsets using unweighted validation BCE.

    For the weighted BCE optimum, logit(true risk) = logit(score) - log(w).
    That correction is an initialization, not an assumption about the trained
    network. Baseline and fitted candidates compete by each head's validation
    log loss; network and normalizer parameters never participate.
    """
    if (probabilities.ndim != 2 or probabilities.shape != labels.shape
            or probabilities.shape[1] != 2 or len(probabilities) == 0):
        raise ValueError("validation probabilities and labels must have shape [samples, 2]")
    if (not torch.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("validation probabilities must be finite; labels must be binary")
    target = labels.detach().to(device=probabilities.device, dtype=torch.float64)
    if ((target.sum(0) == 0) | (target.sum(0) == len(target))).any():
        raise ValueError("each validation head needs both safe and failed samples")
    logits = torch.logit(probabilities.detach().float().clamp(1e-6, 1-1e-6)).double()
    baseline_t = (torch.ones(2, device=logits.device) if initial_temperatures is None
                  else initial_temperatures.detach().to(logits.device)).double()
    baseline_b = (torch.zeros(2, device=logits.device) if initial_biases is None
                  else initial_biases.detach().to(logits.device)).double()
    if (baseline_t.shape != (2,) or baseline_b.shape != (2,)
            or not torch.isfinite(baseline_t).all() or not torch.isfinite(baseline_b).all()
            or (baseline_t <= 0).any()):
        raise ValueError("initial calibration must contain finite positive temperatures and finite biases")
    prior_b = torch.zeros_like(baseline_b)
    if positive_weights is not None:
        weights = positive_weights.detach().to(device=logits.device, dtype=torch.float64)
        if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("positive_weights must contain two finite positive values")
        prior_b = -weights.log()

    def head_losses(t, b):
        return F.binary_cross_entropy_with_logits(logits/t + b, target, reduction="none").mean(0)

    with torch.no_grad():
        baseline_loss = head_losses(baseline_t, baseline_b)
        prior_t = torch.ones_like(baseline_t)
        use_prior = head_losses(prior_t, prior_b) < baseline_loss
        start_t = torch.where(use_prior, prior_t, baseline_t)
        start_b = torch.where(use_prior, prior_b, baseline_b)
    log_t = nn.Parameter(start_t.log().clamp(-7., 7.))
    bias = nn.Parameter(start_b.clone())
    optimizer = torch.optim.LBFGS([log_t, bias], lr=0.5, max_iter=150,
                                 tolerance_grad=1e-9, tolerance_change=1e-12,
                                 line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = head_losses(log_t.exp().clamp(1e-3, 100.), bias.clamp(-30., 30.)).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        candidate_t = torch.stack((baseline_t, start_t, log_t.exp().clamp(1e-3, 100.)))
        candidate_b = torch.stack((baseline_b, start_b, bias.clamp(-30., 30.)))
        losses = torch.stack([head_losses(t, b) for t, b in zip(candidate_t, candidate_b)])
        losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, torch.inf))
        selected = losses.argmin(0)
        heads = torch.arange(2, device=logits.device)
        temperatures = candidate_t[selected, heads].float()
        biases = candidate_b[selected, heads].float()
        if not torch.isfinite(temperatures).all() or not torch.isfinite(biases).all():
            raise RuntimeError("affine calibration failed to produce finite parameters")
        fitted_loss = head_losses(temperatures.double(), biases.double())
    return temperatures.detach(), biases.detach(), {
        "method": "affine_logit", "formula": "sigmoid(logit(raw_risk)/temperature+bias)",
        "fit_split": "validation", "uses_test_labels": False,
        "objective": "unweighted_binary_cross_entropy",
        "samples": len(target), "network_weights_changed": False,
        "baseline_log_loss_per_head": baseline_loss.cpu().tolist(),
        "fitted_log_loss_per_head": fitted_loss.cpu().tolist(),
        "temperatures": temperatures.cpu().tolist(), "biases": biases.cpu().tolist(),
    }
