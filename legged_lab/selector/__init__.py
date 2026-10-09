"""Independent LP/VP reliability selector for the frozen G1 RENet policy.

The package deliberately has no imports from ``legged_lab.envs`` at module
import time.  This keeps the normal training and deployment entry points
unchanged and lets the model, dataset, fault, and runtime components be tested
without starting Isaac Sim.
"""

from .config import (
    CollectionConfig,
    FaultConfig,
    RuntimeConfig,
    SelectorConfig,
    TrainingConfig,
)
from .contracts import DualBranchOutput, SelectorDecision, SelectorInput, SelectorOutput, SensorHealth
from .deployment import OnlineSelectorRuntime, OnlineSelectorStep, ShadowLogWriter
from .features import FeatureHistory, SensorHealthTracker, build_selector_frame
from .faults import FaultKind, SelectorFaultManager
from .model import FeatureNormalizer, SelectorNetwork
from .runtime import ActionBlender, SelectorStateMachine

__all__ = [
    "ActionBlender",
    "CollectionConfig",
    "DualBranchOutput",
    "FaultConfig",
    "FaultKind",
    "FeatureHistory",
    "FeatureNormalizer",
    "OnlineSelectorRuntime",
    "OnlineSelectorStep",
    "RuntimeConfig",
    "SelectorConfig",
    "SelectorDecision",
    "SelectorFaultManager",
    "SelectorInput",
    "SelectorOutput",
    "SelectorNetwork",
    "SelectorStateMachine",
    "SensorHealth",
    "SensorHealthTracker",
    "ShadowLogWriter",
    "TrainingConfig",
    "build_selector_frame",
]
