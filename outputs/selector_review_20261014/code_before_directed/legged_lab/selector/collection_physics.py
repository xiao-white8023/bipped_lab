"""Apply collection-only physics options to the actual SimulationContext cfg."""

from __future__ import annotations

import importlib


def collection_physics_settings(env) -> dict:
    sim_cfg = getattr(getattr(env, "sim", None), "cfg", None)
    physx = getattr(sim_cfg, "physx", None)
    return {
        "device": str(getattr(env, "device", "unknown")),
        "physics_dt": getattr(env, "physics_dt", None),
        "control_dt": getattr(env, "step_dt", None),
        "enable_enhanced_determinism": getattr(physx, "enable_enhanced_determinism", None),
        "solver_type": getattr(physx, "solver_type", None),
    }


def create_collection_environment(env_class, cfg, headless):
    """Scope the override to this constructor; leave training source intact.

    G1RENetEnv constructs a fresh PhysxCfg and copies only the patch capacity,
    so setting cfg.sim.physx.enable_enhanced_determinism alone has no effect.
    The intercepted context receives the actual config before scene creation.
    """
    module = importlib.import_module(env_class.__module__)
    original = getattr(module, "SimulationContext", None)
    if not callable(original):
        raise RuntimeError("collection environment must expose its SimulationContext constructor")
    configured = []

    def construct(sim_cfg, *args, **kwargs):
        physx = getattr(sim_cfg, "physx", None)
        if physx is None or not hasattr(physx, "enable_enhanced_determinism"):
            raise RuntimeError("actual SimulationContext lacks the requested PhysX determinism option")
        physx.enable_enhanced_determinism = True
        configured.append(sim_cfg)
        return original(sim_cfg, *args, **kwargs)

    module.SimulationContext = construct
    try:
        env = env_class(cfg, headless)
    finally:
        module.SimulationContext = original
    if len(configured) != 1:
        raise RuntimeError(f"expected one collection SimulationContext, observed {len(configured)}")
    if collection_physics_settings(env)["enable_enhanced_determinism"] is not True:
        raise RuntimeError("actual collection simulation did not retain enhanced determinism")
    return env
