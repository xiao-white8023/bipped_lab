"""Independent random streams for vectorized counterfactual collection.

Counters belong to environment IDs, not to batches. Resetting, sampling or
resampling row B never advances row A. No process-wide torch APIs are patched.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any

import torch


class PerEnvironmentRNG:
    """Counter-based uniform/normal draws; state is one int64 per environment."""

    def __init__(self, num_envs: int, seed: int, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.seed = int(seed) & 0xFFFFFFFF
        self.counter = torch.zeros(num_envs, dtype=torch.int64, device=self.device)

    def get_state(self) -> torch.Tensor:
        return self.counter.clone()

    def set_state(self, state: torch.Tensor) -> None:
        if state.shape != self.counter.shape:
            raise ValueError("per-environment RNG state shape changed")
        self.counter.copy_(state)

    def uniform(self, ids: torch.Tensor, shape: tuple[int, ...] = ()) -> torch.Tensor:
        ids = ids.to(self.device, torch.long).reshape(-1)
        width = math.prod(shape)
        elements = torch.arange(width, device=self.device, dtype=torch.int64)[None]
        x = (
            self.seed
            + ids[:, None] * 0x9E3779B9
            + self.counter[ids, None] * 0x85EBCA6B
            + elements * 0xC2B2AE35
        ) & 0xFFFFFFFF
        for _ in range(2):
            x = ((x ^ (x >> 16)) * 0x045D9F3B) & 0xFFFFFFFF
        x = x ^ (x >> 16)
        # Keep both endpoints excluded after conversion to float32.
        value = ((x & 0xFFFFFF).float() + 0.5) / 16777216.0
        value = value.clamp(2.0**-24, 1.0 - 2.0**-24)
        self.counter[ids] += 1
        return value.reshape(len(ids), *shape)

    def normal(self, ids: torch.Tensor, shape: tuple[int, ...] = ()) -> torch.Tensor:
        u = self.uniform(ids, shape)
        v = self.uniform(ids, shape)
        return torch.sqrt(-2.0 * u.log()) * torch.cos(2.0 * math.pi * v)

    def randint(
        self, low: int, high: int, ids: torch.Tensor, shape: tuple[int, ...] = ()
    ) -> torch.Tensor:
        if high <= low:
            raise ValueError("randint requires high > low")
        return (self.uniform(ids, shape) * (high - low)).long().clamp_max(
            high - low - 1
        ) + low

    def categorical(
        self, probabilities: tuple[float, ...], ids: torch.Tensor
    ) -> torch.Tensor:
        bounds = torch.tensor(probabilities, device=self.device).cumsum(0)
        return (
            (self.uniform(ids)[:, None] > bounds[None])
            .sum(-1)
            .clamp_max(len(probabilities) - 1)
        )


class EnvironmentRandomness:
    """Named row-local streams, plus isolated seeds for existing Isaac helpers."""

    def __init__(self, num_envs: int, seed: int, device: str | torch.device):
        self.num_envs, self.seed, self.device = (
            num_envs,
            int(seed),
            torch.device(device),
        )
        self.streams: dict[str, PerEnvironmentRNG] = {}

    def _stream(self, name: str) -> PerEnvironmentRNG:
        if name not in self.streams:
            salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            self.streams[name] = PerEnvironmentRNG(
                self.num_envs, self.seed ^ salt, self.device
            )
        return self.streams[name]

    def ids(self, ids: Any = None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device)
        if ids is None:
            return all_ids
        if isinstance(ids, slice):
            return all_ids[ids]
        return torch.as_tensor(ids, dtype=torch.long, device=self.device).reshape(-1)

    def uniform(self, ids: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
        return self._stream(name).uniform(self.ids(ids), shape)

    def normal(self, ids: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
        return self._stream(name).normal(self.ids(ids), shape)

    def randint(self, low: int, high: int, ids: Any, name: str) -> torch.Tensor:
        return self._stream(name).randint(low, high, self.ids(ids))

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: stream.get_state() for name, stream in self.streams.items()}

    def restore(self, state: dict[str, torch.Tensor]) -> None:
        # A fork may have created a stream absent at the candidate boundary.
        for name in set(self.streams) - set(state):
            del self.streams[name]
        for name, value in state.items():
            self._stream(name).set_state(value)

    @contextmanager
    def single_env_seed(self, env_id: int, name: str):
        ids = torch.tensor([env_id], device=self.device)
        seed = int(self._stream(name).randint(0, 2**31 - 1, ids).item())
        cuda_devices = (
            []
            if self.device.type != "cuda"
            else [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            for device in cuda_devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            yield


def bind_environment_randomness(env: Any, rng: EnvironmentRandomness) -> None:
    """Isolate command/reset/push draws without changing PPO entry points.

    Isaac helper methods are installed only in the collection process. Sensor
    and proprio noise use vectorized named streams in the environment/helper
    code. Unknown interval layouts fail explicitly instead of claiming parity.
    """
    env.selector_randomness = rng
    if getattr(env, "_selector_randomness_bound", False):
        raise RuntimeError("environment randomness is already bound to a collector")
    env._selector_randomness_bound = True
    if getattr(env, "pgtt_noise", None) is not None:
        env.pgtt_noise.selector_randomness = rng
    command = getattr(env, "command_generator", None)
    if command is not None and hasattr(command, "_resample"):
        original = command._resample

        def resample(ids):
            for env_id in rng.ids(ids).tolist():
                with rng.single_env_seed(env_id, "command_resample"):
                    original(torch.tensor([env_id], device=rng.device))

        command._resample = resample
    elif command is not None:
        raise RuntimeError(
            "selector collection requires CommandTerm._resample for row-local RNG"
        )

    manager = getattr(env, "event_manager", None)
    if manager is None:
        return
    modes = getattr(manager, "_mode_term_cfgs", None)
    if not isinstance(modes, dict):
        raise RuntimeError(
            "unsupported Isaac EventManager layout: missing _mode_term_cfgs"
        )
    for mode in ("reset", "interval"):
        for index, cfg in enumerate(modes.get(mode, [])):
            original_func = cfg.func

            def event(
                env_arg,
                env_ids=None,
                _func=original_func,
                _name=f"event_{mode}_{index}",
                **kwargs,
            ):
                for env_id in rng.ids(env_ids).tolist():
                    with rng.single_env_seed(env_id, _name):
                        _func(
                            env_arg, torch.tensor([env_id], device=rng.device), **kwargs
                        )

            if hasattr(original_func, "reset"):

                def reset_term(
                    env_ids=None,
                    _func=original_func,
                    _name=f"event_reset_{mode}_{index}",
                ):
                    for env_id in rng.ids(env_ids).tolist():
                        with rng.single_env_seed(env_id, _name):
                            _func.reset(
                                env_ids=torch.tensor([env_id], device=rng.device)
                            )

                event.reset = reset_term
            cfg.func = event
    interval = modes.get("interval", [])
    if not interval:
        return
    timers = getattr(manager, "_interval_term_time_left", None)
    if timers is None or len(timers) != len(interval):
        raise RuntimeError("unsupported Isaac interval timer layout")
    if any(getattr(cfg, "is_global_time", False) for cfg in interval):
        raise RuntimeError(
            "selector collection supports row-local interval events only"
        )
    if any(timer.numel() != env.num_envs for timer in timers):
        raise RuntimeError("interval timers must have one value per environment")
    original_apply = manager.apply

    def apply(mode, *args, **kwargs):
        expired = []
        if mode == "interval":
            dt = kwargs.get("dt")
            if dt is None:
                raise ValueError("interval randomness needs an explicit dt")
            expired = [(timer - dt < 1.0e-6).nonzero().flatten() for timer in timers]
        result = original_apply(mode, *args, **kwargs)
        if mode == "interval":
            for index, (cfg, timer, ids) in enumerate(zip(interval, timers, expired)):
                low, high = cfg.interval_range_s
                timer[ids] = low + (high - low) * rng.uniform(
                    ids, (), f"interval_timer_{index}"
                )
        return result

    manager.apply = apply
    if hasattr(manager, "reset"):
        original_reset = manager.reset

        def reset(env_ids=None):
            result = original_reset(env_ids)
            ids = rng.ids(env_ids)
            for index, (cfg, timer) in enumerate(zip(interval, timers)):
                low, high = cfg.interval_range_s
                timer[ids] = low + (high - low) * rng.uniform(
                    ids, (), f"interval_timer_{index}"
                )
            return result

        manager.reset = reset
