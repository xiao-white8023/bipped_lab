# 必须放在文件最顶部，启用注解延迟解析
from __future__ import annotations
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from dataclasses import MISSING

# 仅静态类型检查/IDE补全时导入，运行时不执行，彻底打破循环
if TYPE_CHECKING:
    from legged_lab.utils.height_cammand.height_cammand import UniformHeightCommand


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    # 用字符串全路径代替直接类引用，Isaac Lab管理器会自动反射导入
    class_type: type[UniformHeightCommand] = "legged_lab.utils.height_cammand.height_cammand.UniformHeightCommand"
    asset_name: str = MISSING
    rel_standing_envs: float = 0.0
    stand_height:float = MISSING

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING  # 目标高度

    ranges: Ranges = MISSING