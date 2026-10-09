"""Small calibrated recurrent risk model used by the LP/VP selector."""

from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .config import SelectorConfig


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


class SelectorNetwork(nn.Module):
    """Frame MLP + GRU with monotone short/long failure probabilities."""

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

    def raw_outputs(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("selector history must have shape [batch, time, frame_dim]")
        if history.shape[1] != self.cfg.history_frames or history.shape[2] != self.cfg.frame_dim:
            raise ValueError(
                f"selector history must have shape [batch, {self.cfg.history_frames}, {self.cfg.frame_dim}]"
            )
        encoded = self.frame_encoder(history)
        _, hidden = self.temporal(encoded)
        return self.head(hidden[-1])

    @staticmethod
    def probabilities_from_raw(raw: torch.Tensor) -> torch.Tensor:
        """Convert two short risks and two conditional increments to four risks."""

        # Probability arithmetic must retain the FP32 clamp margin under AMP.
        raw = raw.float()
        lp_short = torch.sigmoid(raw[:, 0])
        lp_increment = torch.sigmoid(raw[:, 1])
        vp_short = torch.sigmoid(raw[:, 2])
        vp_increment = torch.sigmoid(raw[:, 3])
        lp_long = lp_short + (1.0 - lp_short) * lp_increment
        vp_long = vp_short + (1.0 - vp_short) * vp_increment
        return torch.stack((lp_short, lp_long, vp_short, vp_long), dim=-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.probabilities_from_raw(self.raw_outputs(history))


class CalibratedSelector(nn.Module):
    """Exportable normalizer, selector, and per-output temperature scaling."""

    def __init__(
        self,
        model: SelectorNetwork,
        normalizer: FeatureNormalizer | None = None,
        temperatures: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer or FeatureNormalizer(model.cfg.frame_dim)
        if temperatures is None:
            temperatures = torch.ones(4)
        if temperatures.shape != (4,):
            raise ValueError("temperatures must have shape [4]")
        self.register_buffer("temperatures", temperatures.float().clamp_min(1.0e-3))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        probabilities = self.model(self.normalizer(history))
        logits = torch.logit(probabilities.clamp(1.0e-6, 1.0 - 1.0e-6))
        calibrated = torch.sigmoid(logits / self.temperatures)
        # Independent temperature values are useful for calibration but can
        # otherwise invert a short/long pair.  Restore the model contract after
        # calibration without changing either short-horizon output.
        lp_long = torch.maximum(calibrated[:, 0], calibrated[:, 1])
        vp_long = torch.maximum(calibrated[:, 2], calibrated[:, 3])
        return torch.stack(
            (calibrated[:, 0], lp_long, calibrated[:, 2], vp_long), dim=-1
        )


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
    if probabilities.shape != targets.shape or probabilities.shape[-1] != 4:
        raise ValueError("probabilities and targets must have matching [batch, 4] shapes")
    if positive_weights.shape != (4,):
        raise ValueError("positive_weights must have shape [4]")
    probabilities = probabilities.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    targets = targets.to(probabilities.dtype)
    loss = -(
        positive_weights[None] * targets * torch.log(probabilities)
        + (1.0 - targets) * torch.log1p(-probabilities)
    )
    return loss.mean()


@torch.no_grad()
def positive_class_weights(labels: torch.Tensor, maximum: float = 100.0) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError("labels must have shape [samples, 4]")
    positive = labels.float().sum(dim=0)
    negative = labels.shape[0] - positive
    return (negative / positive.clamp_min(1.0)).clamp(1.0, maximum)


def fit_temperatures(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Fit four positive scalar temperatures on a validation split."""

    if probabilities.shape != labels.shape or probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("validation probabilities and labels must have shape [samples, 4]")
    logits = torch.logit(probabilities.detach().clamp(1.0e-6, 1.0 - 1.0e-6))
    log_temperature = nn.Parameter(torch.zeros(4, device=probabilities.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1.0e-3, 100.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(1.0e-3, 100.0)
