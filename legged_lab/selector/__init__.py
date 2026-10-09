# 中文导读
# 文件用途：提供独立 LP/VP 可靠性选择器的统一 Python 导入入口。
# 使用位置：外部代码通过 legged_lab.selector 导入配置、数据类型、模型和在线运行组件。
# 说明：本文件通过 __all__ 列出公开接口，类与函数的具体实现位于同目录的各模块。
# 说明：本文件没有自行定义类或函数；导入包不会启动 Isaac Sim，也不会启动 RENet 训练。
# 中文导读结束

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
