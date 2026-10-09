# 中文导读
# 文件用途：统计地形难度与类型分布，并据此固定正式采集环境的地形安排。
# 使用位置：collect_selector_data.py 的地形预采样、主数据采集和难度 7～9 的泛化测试采集。
# 说明：terrain_distribution.json 记录场景统计；selector 样本由 collector 的受扰成对 rollout 生成。
# 说明：reference_mean 仅记录参考值；实际采样使用观测计数，不强制难度均值等于参考值。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - TerrainDistribution：保存难度与类型计数矩阵、训练最高难度和参考均值。
# - TerrainDistribution.empty：建立指定难度数和类型数的零计数统计对象。
# - TerrainDistribution.observe：累计合法类型且难度在训练范围内的地形观测。
# - TerrainDistribution.sample_count：属性访问器，返回累计的地形观测总数。
# - TerrainDistribution.mean_level：属性访问器，计算按观测次数加权的难度均值。
# - TerrainDistribution.probabilities：将训练范围内的计数归一化为地形抽样概率，拒绝空分布。
# - TerrainDistribution.sample：按观测到的难度与类型联合分布为各环境抽样。
# - TerrainDistribution.sample_ood：保留观测到的类型比例，在指定的高难度集合中抽取测试场景。
# - TerrainDistribution.save：将计数、范围、均值和观测总数保存为 JSON。
# - TerrainDistribution.load：读取 JSON 并在指定设备上重建统计张量。
# - apply_frozen_distribution：为环境设置抽取的地形难度、类型和起点，再 reset 所有机器人。
# 中文导读结束

"""Empirical terrain distribution capture and frozen sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


@dataclass
class TerrainDistribution:
    counts: torch.Tensor
    max_train_level: int = 6
    reference_mean: float = 5.27

    @classmethod
    def empty(cls, levels: int, terrain_types: int, device: str | torch.device = "cpu") -> "TerrainDistribution":
        return cls(torch.zeros(levels, terrain_types, dtype=torch.long, device=device))

    def observe(self, levels: torch.Tensor, terrain_types: torch.Tensor) -> None:
        levels = levels.reshape(-1).long()
        terrain_types = terrain_types.reshape(-1).long()
        valid = (
            (levels >= 0)
            & (levels <= self.max_train_level)
            & (terrain_types >= 0)
            & (terrain_types < self.counts.shape[1])
        )
        flattened = levels[valid] * self.counts.shape[1] + terrain_types[valid]
        update = torch.bincount(flattened, minlength=self.counts.numel()).reshape_as(self.counts)
        self.counts += update.to(self.counts.device)

    @property
    def sample_count(self) -> int:
        return int(self.counts.sum().item())

    @property
    def mean_level(self) -> float:
        total = self.counts.sum().clamp_min(1)
        levels = torch.arange(self.counts.shape[0], device=self.counts.device)[:, None]
        return float((levels * self.counts).sum() / total)

    def probabilities(self) -> torch.Tensor:
        masked = self.counts.clone().float()
        if masked.shape[0] > self.max_train_level + 1:
            masked[self.max_train_level + 1 :] = 0
        if masked.sum() <= 0:
            raise RuntimeError("terrain distribution has no level 0-6 samples")
        return masked.flatten() / masked.sum()

    def sample(
        self,
        num_envs: int,
        *,
        device: str | torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.probabilities().to(device)
        selected = torch.multinomial(probabilities, num_envs, replacement=True, generator=generator)
        terrain_types = selected % self.counts.shape[1]
        levels = torch.div(selected, self.counts.shape[1], rounding_mode="floor")
        return levels.long(), terrain_types.long()

    def sample_ood(
        self,
        num_envs: int,
        *,
        levels: tuple[int, ...] = (7, 8, 9),
        device: str | torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample OOD levels while retaining the observed terrain-type marginal."""

        available = tuple(level for level in levels if 0 <= level < self.counts.shape[0])
        if not available:
            raise RuntimeError("terrain generator does not contain requested OOD levels")
        type_counts = self.counts[: self.max_train_level + 1].sum(dim=0).float()
        if type_counts.sum() <= 0:
            raise RuntimeError("terrain distribution has no terrain-type observations")
        type_probability = (type_counts / type_counts.sum()).to(device)
        terrain_types = torch.multinomial(
            type_probability, num_envs, replacement=True, generator=generator
        )
        choices = torch.tensor(available, device=device, dtype=torch.long)
        selected = torch.randint(
            0, len(available), (num_envs,), device=device, generator=generator
        )
        return choices[selected], terrain_types.long()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "counts": self.counts.cpu().tolist(),
            "max_train_level": self.max_train_level,
            "reference_mean": self.reference_mean,
            "observed_mean": self.mean_level,
            "sample_count": self.sample_count,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "TerrainDistribution":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            counts=torch.tensor(payload["counts"], dtype=torch.long, device=device),
            max_train_level=int(payload["max_train_level"]),
            reference_mean=float(payload.get("reference_mean", 5.27)),
        )


def apply_frozen_distribution(
    env: Any,
    distribution: TerrainDistribution,
    *,
    generator: torch.Generator,
    ood_levels: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign level/type pairs and reset all robots at the matching origins."""

    terrain = env.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        raise RuntimeError("empirical terrain sampling requires generated terrain origins")
    if ood_levels is None:
        levels, terrain_types = distribution.sample(
            env.num_envs, device=env.device, generator=generator
        )
    else:
        levels, terrain_types = distribution.sample_ood(
            env.num_envs,
            levels=ood_levels,
            device=env.device,
            generator=generator,
        )
    if levels.max() >= terrain.terrain_origins.shape[0]:
        raise RuntimeError("terrain histogram contains a level absent from this terrain generator")
    if terrain_types.max() >= terrain.terrain_origins.shape[1]:
        raise RuntimeError("terrain histogram contains a type absent from this terrain generator")
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(terrain_types)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, terrain_types])
    env.scene.env_origins.copy_(terrain.env_origins)
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)
    return levels, terrain_types
