from __future__ import annotations

import logging
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.sim.views import XformPrimView

from legged_lab.utils.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg

# 导入日志记录器
logger = logging.getLogger(__name__)


class GroupedRayCaster(MultiMeshRayCaster):
    """分组光线投射传感器读取多个IsaacSim prim路径，并在投射光线前持续更新网格位置。"""

    cfg: GroupedRayCasterCfg
    """ 配置参数 """

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)  # 调用父类的初始化方法 并将配置类传给这个初始化方法

    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()

        # 我们创建一个展平的网格ID张量，与展平的网格变换一一对应。
        # # self._mesh_positions_w 父类中的是一个存储了所有网格世界坐标的张量，它的形状通常是 (环境总数, 每个环境的网格数, 3）
        total_meshes_per_env = self._mesh_positions_w.shape[1] 
        # 创建一个全为0的张量，形状为 (self._num_envs, total_meshes_per_env)
        mesh_wp_ids_tensor = torch.zeros(
            (self._num_envs, total_meshes_per_env),
            dtype=torch.int64,
            device=self._device,
        )


        '''

        self._raycast_targets_cfg 本质是标准化后的射线投射目标配置列表，它既兼容了简单的字符串路径配置，也支持通过 RaycastTargetCfg 定义的高级参数，最终驱动后续的网格加载、去重、位姿追踪等核心逻辑。
        self._raycast_targets_cfg 是之前初始化好的配置列表，每个元素 target_cfg 是一个 RaycastTargetCfg 对象，包含：
        prim_expr：目标 prim 的路径表达式（支持正则）。
        is_shared：是否为共享网格。
        track_mesh_transforms：是否追踪位姿变化。

        '''
        mesh_idx = 0
        for target_cfg in self._raycast_targets_cfg:  # 射线要检测的网格
            prims = sim_utils.find_matching_prims(target_cfg.prim_expr)  # 找到所有环境中的对应的目标网格  返回一个列表 举例：如果有 env_0、env_1 两个环境，这行会返回 [env_0的左腿Prim, env_1的左腿Prim]。
            ids = []
            for prim in prims:  # 遍历所有环境的目标网格
                prim_path = prim.GetPath().pathString  # # 获取当前网格的绝对路径，例如 "/World/envs/env_25/Robot/left_leg"
                # 【关键步骤】利用正则表达式，把路径中的 "env_数字" 强制替换为 "env_0"
                # "/World/envs/env_25/Robot/left_leg" -> "/World/envs/env_0/Robot/left_leg"
                prim_path_ = regex.sub(r"env_\d+", "env_0", prim_path)
                assert prim_path_ in GroupedRayCaster.meshes, (
                    f"在网格缓存中未找到prim路径为 {prim_path}（转换为 {prim_path_}）的网格"
                    f" {GroupedRayCaster.meshes.keys()}"
                )
                ids.append(GroupedRayCaster.meshes[prim_path_].id)

            ids_tensor = torch.tensor(ids, device=self._device, dtype=torch.int64)
            '''
            这个是父类执行完毕后self._num_meshes_per_env的类型
            {
                "/World/Ground": 10, 
                "/World/envs/env_.*/Robot/LF_leg/visuals": 3,
                "/World/envs/env_.*/Robot/RF_leg/visuals": 3
            }

            键 (Key)：字符串 (str)。代表射线投射目标的 Prim 路径表达式（即你在配置中传入的 target_cfg.prim_expr，例如 "/World/envs/env_.*/Robot/LF_leg/visuals"）。
            值 (Value)：整数 (int)。代表该目标路径在**单个环境（per environment）**中实际包含的底层物理网格（Mesh）的数量。

            '''
            count = self._num_meshes_per_env[target_cfg.prim_expr]

            if len(ids) == 1:  # 说明是全局网格 例如地面
                mesh_wp_ids_tensor[:, mesh_idx] = ids_tensor[0]
            elif len(ids) == count:
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.unsqueeze(0)
            elif len(ids) == self._num_envs * count:
                mesh_wp_ids_tensor[:, mesh_idx : mesh_idx + count] = ids_tensor.view(self._num_envs, count)
            else:
                logger.warning(f"{target_cfg.prim_expr} 的网格数量不匹配")

            mesh_idx += count

        self._mesh_wp_ids = mesh_wp_ids_tensor.flatten()  # 展平为一维的 环境数 × 单个环境的总网格数

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # 创建缓冲区以存储光线碰撞组
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        '''
        物理意义：给所有的光线发“身份证”。如果环境 0 里有 10 根射线，环境 1 里有 10 根射线，GPU 在算的时候得知道某根射线到底该去和哪个环境的物体求交。

        张量变化：

        arange 得到 [0, 1, 2, ... num_envs-1]。

         unsqueeze(1) 变成列向量：

        [[0],
        [1],
        [2]]

        repeat(1, self.num_rays) 把它横向复制 num_rays 次。最终形状是 (num_envs, num_rays)：

        [[0, 0, 0, ...],  # 环境 0 的所有射线，碰撞组都是 0
        [1, 1, 1, ...],  # 环境 1 的所有射线，碰撞组都是 1
        [2, 2, 2, ...]]

        作用：告诉 Warp 内核：“当你处理 _ray_collision_groups[i][j] 这根射线时，它只能和第 i 组（即第 i 个环境）里的网格发生碰撞，别去撞隔壁环境的网格。”
        '''
        self._ray_collision_groups = (
            torch.arange(self._num_envs, dtype=torch.int32, device=self._device).unsqueeze(1).repeat(1, self.num_rays)
        )

        '''
        这是最核心的部分，目的是给每个环境的每个网格分配一个全局唯一索引，方便 GPU 快速定位。
        创建一个全为-1的张量   形状是(环境数，每个环境的总网格数)
        '''
        _mesh_idxs_for_group = torch.ones(
            (self._mesh_positions_w.shape[0], self._mesh_positions_w.shape[1]),
            dtype=torch.int32,
            device=self._device,
        ).fill_(-1)

        '''
        没关系，这段代码用了 PyTorch 的**张量广播（Broadcasting）**机制，没有接触过的话确实像天书。

我们抛开抽象的概念，直接用一个具体的数字例子来一步步推演，你瞬间就能明白。
假设一个场景（设定数字）

    环境总数 (num_envs) = 3个（我们在跑3个并行的仿真环境：环境0、环境1、环境2）。

    单个环境的网格总数 (total_meshes) = 5个（每个环境里总共有5个网格点）。

        这意味着，在GPU底层，所有网格排成了一条长度为 15 (3x5) 的一维长数组，下标是 0 到 14。

            环境0的网格下标：0, 1, 2, 3, 4

            环境1的网格下标：5, 6, 7, 8, 9

            环境2的网格下标：10, 11, 12, 13, 14

    机器人的组成 (_raycast_targets_cfg)：假设由2个部位组成：

        第一个部位（比如“身体”）：由 2 个网格组成。

        第二个部位（比如“手臂”）：由 3 个网格组成。（2+3刚好等于单环境总数5）。

现在我们要填满一个大小为 (3, 5) 的表格 _mesh_idxs_for_group。
第 1 次循环：处理“身体”（2个网格）

此时，mesh_idx = 0，count = 2。我们来看那行神奇的公式是怎么算的：

第一项：计算每个环境的基准起点
torch.arange(3).unsqueeze(1) * 5
这会生成一个列向量 (3, 1)：
Plaintext

[[0],   (环境0的起点)
 [5],   (环境1的起点)
 [10]]  (环境2的起点)

第二项：计算“身体”内部的偏移
torch.arange(2).unsqueeze(0)
这会生成一个行向量 (1, 2)：
Plaintext

[[0, 1]]  (身体有2个网格，分别是第0个和第1个)

第三项：加上 mesh_idx
此时 mesh_idx = 0。

相加（奇迹发生的地方：PyTorch 广播机制）
列向量 (3, 1) 加上 行向量 (1, 2)，PyTorch 会自动把它们扩展成 (3, 2) 的矩阵相加：
Plaintext

  [[0],        [[0, 1],        [[ 0,  1],   <-- 环境0的身体，在全局数组的下标是 0 和 1
   [5],    +    [0, 1],    =    [ 5,  6],   <-- 环境1的身体，在全局数组的下标是 5 和 6
   [10]]        [0, 1]]         [10, 11]]   <-- 环境2的身体，在全局数组的下标是 10 和 11

算出来的这个 indices 矩阵，直接塞进表格的第0到第1列：
_mesh_idxs_for_group[:, 0 : 2] = indices

循环最后：mesh_idx += count
mesh_idx 变成了 0 + 2 = 2。告诉下一次循环：“前面2列我已经填完了，你从第2列开始填”。
第 2 次循环：处理“手臂”（3个网格）

此时，mesh_idx = 2，count = 3。

第一项：基准起点不变
Plaintext

[[0], 
 [5], 
 [10]]

第二项：手臂内部的偏移
torch.arange(3).unsqueeze(0) 生成 (1, 3) 的行向量：
Plaintext

[[0, 1, 2]]

第三项：加上 mesh_idx
加上上一轮留下的 mesh_idx = 2。相当于偏移量变成了 [[0+2, 1+2, 2+2]] = [[2, 3, 4]]。
(这非常合理，因为每个环境的 0、1 号网格已经被身体占了，手臂只能是 2、3、4 号网格)

相加：
Plaintext

  [[0],        [[2, 3, 4],        [[ 2,  3,  4],   <-- 环境0的手臂，在全局的下标是 2, 3, 4
   [5],    +    [2, 3, 4],    =    [ 7,  8,  9],   <-- 环境1的手臂，在全局的下标是 7, 8, 9
   [10]]        [2, 3, 4]]         [12, 13, 14]]   <-- 环境2的手臂，在全局的下标是 12, 13, 14

算出来的 indices 矩阵，直接塞进表格的第2到第4列：
_mesh_idxs_for_group[:, 2 : 5] = indices
最终结果

两次循环跑完，刚才那个装满 -1 的 (3, 5) 表格变成了这样：
Plaintext

[[ 0,  1,   2,  3,  4],    <- 环境 0 专属的5个网格全局下标
 [ 5,  6,   7,  8,  9],    <- 环境 1 专属的5个网格全局下标
 [10, 11,  12, 13, 14]]    <- 环境 2 专属的5个网格全局下标

总结：
这段代码的精妙之处在于，它完全没有用 for 循环去遍历几千个环境。而是利用矩阵相加（列向量 + 行向量），一次性把所有环境在这一个部位上的全局下标全部算了出来，然后像拼图一样，一块一块（一列一列）地拼到大表格里。
        '''
        mesh_idx = 0
        total_meshes = self._mesh_positions_w.shape[1]  # 单个环境的所有的网格总数
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            # 获取当前目标在单环境中的网格数
            count = self._num_meshes_per_env[target_cfg.prim_expr]

            indices = (
                torch.arange(self._num_envs, device=self._device).unsqueeze(1) * total_meshes
                + torch.arange(count, device=self._device).unsqueeze(0)
                + mesh_idx
            )
            _mesh_idxs_for_group[:, mesh_idx : mesh_idx + count] = indices.int()
            mesh_idx += count
        self._mesh_idxs_for_group = _mesh_idxs_for_group.flatten(0, 1) 

        _meah_idxs_slice_for_group = torch.arange(self._num_envs + 1, dtype=torch.int32, device=self._device)
        _meah_idxs_slice_for_group *= self._mesh_positions_w.shape[1]
        self._meah_idxs_slice_for_group = _meah_idxs_slice_for_group  # (num_envs + 1)

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        """
        更新给定环境ID的网格变换。

        参数：
            env_ids：需要更新网格变换的环境ID。
        """
        # 更新网格位置和旋转
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # 更新目标网格的位置
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # 网格不是全局的，即每个环境有不同的网格
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w  # (w, x, y, z)
            mesh_idx += count

    def _get_mesh_transforms_and_inv_transforms(self):
        """获取给定环境ID的网格变换和逆变换。"""
        mesh_transforms = torch.concatenate(
            [self._mesh_positions_w, self._mesh_orientations_w],
            dim=-1,
        ).reshape(
            -1, 7
        )  # (num_envs * (global_meshes + local_meshes_per_env), 7) # (px, py, pz, qw, qx, qy, qz)
        # 计算逆变换
        # inv(T) = (inv(q) * -p, inv(q))
        inv_q = math_utils.quat_inv(self._mesh_orientations_w)
        inv_p = math_utils.quat_apply(inv_q, -self._mesh_positions_w)
        mesh_inv_transforms = torch.concatenate(
            [inv_p, inv_q],
            dim=-1,
        ).reshape(
            -1, 7
        )  # (num_envs * (global_meshes + local_meshes_per_env), 7) # (px, py, pz, qw, qx, qy, qz)
        return mesh_transforms, mesh_inv_transforms

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """使用当前网格位置和方向更新光线投射器缓冲区。同时更新给定环境ID（即碰撞组ID）上的网格点。

        参数：
            env_ids：需要更新缓冲区的环境ID。
        """
        self._update_ray_infos(env_ids)
        self._update_mesh_transforms(env_ids)

        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        mesh_wp = [i for i in GroupedRayCaster.meshes.values()][0]
        self._data.ray_hits_w[env_ids], _, _, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=mesh_transforms,
            mesh_inv_transforms=mesh_inv_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=self._ray_starts_w[env_ids],
            ray_directions=self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
        )
