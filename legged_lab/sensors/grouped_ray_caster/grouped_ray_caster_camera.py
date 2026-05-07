from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import isaacsim.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import CameraData
from isaaclab.sensors.ray_caster import RayCasterCamera
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view

from legged_lab.utils.raycast import raycast_mesh_grouped

from . import GroupedRayCaster

if TYPE_CHECKING:
    from .grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg

# import logger
logger = logging.getLogger(__name__)


class GroupedRayCasterCamera(RayCasterCamera, GroupedRayCaster):
    """Grouped ray-cast camera sensor."""

    cfg: GroupedRayCasterCameraCfg

    """The configuration parameters."""
    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GroupedRayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the grouped-ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._data = CameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Grouped-Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Implementations.
    """

    def _initialize_warp_meshes(self):
        GroupedRayCaster._initialize_warp_meshes(self)

    def _initialize_rays_impl(self):
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = self.ray_directions.shape[1]
        # create buffer to store ray hits
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)

        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._create_ray_collision_groups()

    def _update_ray_infos(self, env_ids: Sequence[int]):
        """Updates the ray information buffers."""

        # compute poses from current view
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        self._data.quat_w_ros[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        '''
        # ---------------- 步骤 6：保存结果 ----------------
        # 把这一帧算好的所有射线的世界起点和方向缓存下来。
        # 这两个变量随后就会被传给我们在上一个问题里讲到的 raycast_mesh_grouped (物理底层) 去做真正的光线追踪碰撞检测。
        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w
        '''
        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    '''
    它的主要作用是：在每一个仿真步（Step）中，更新相机的位置和姿态，发射射线（Raycast）与环境中的网格（Mesh）进行物理碰撞检测，并将碰撞结果（距离、法线等）转化为我们常见的相机图像数据（如深度图、法线图）。
    '''
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        '''
        # 1. 更新射线信息：根据当前相机的位置和旋转（姿态），计算出每一条射线在世界坐标系下的起点（ray_starts_w）和方向（ray_directions_w）。
        '''
        self._update_ray_infos(env_ids)
        '''
        环境中可能有移动的物体或地形，这一步更新所有参与碰撞检测的网格（Mesh）在世界坐标系下的当前位姿（平移和旋转
        '''
        self._update_mesh_transforms(env_ids)

        mesh_transforms, mesh_inv_transforms = self._get_mesh_transforms_and_inv_transforms()

        mesh_wp = [i for i in GroupedRayCaster.meshes.values()][0]
        self.ray_hits_w, ray_depth, ray_normal, _, _ = raycast_mesh_grouped(
            mesh_wp_device=mesh_wp.device,             # 指定运行所在的 GPU 设备
            mesh_wp_ids=self._mesh_wp_ids,             # 参与计算的 Mesh 的底层 ID
            mesh_transforms=mesh_transforms,           # Mesh 的世界坐标变换
            mesh_inv_transforms=mesh_inv_transforms,   # Mesh 的逆变换
            ray_group_ids=self._ray_collision_groups[env_ids], # 碰撞组别（决定哪些射线只能和哪些物体碰撞，比如过滤掉机器人自己）
            mesh_idxs_for_group=self._mesh_idxs_for_group,     # 每组对应的网格索引
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group, # 内存切片加速
            ray_starts=self._ray_starts_w[env_ids],    # 所有射线的起点 (N, num_rays, 3)
            ray_directions=self._ray_directions_w[env_ids], # 所有射线的方向 (N, num_rays, 3)
            
            # 最大探测距离乘以2的原因：
            # 这里算的是欧氏距离（也就是斜边）。如果相机视场角(FOV)很大，画面边缘的射线斜边会比垂直深度长很多。
            # 为了防止在转算成垂直深度(distance_to_image_plane)时被错误截断，底层物理探测的距离要放宽一倍。
            max_dist=self.cfg.max_distance * 2,  
            min_dist=self.cfg.min_distance,            # 最小探测距离（类似相机的近裁剪面 Near Clipping Plane）
            return_distance=True,                      # 要求返回射线命中的距离 (ray_depth)
            return_normal=True,                        # 要求返回命中点表面的法线向量 (ray_normal)
        )
        # 断言确保 GPU 正确返回了深度和法线数据
        assert ray_depth is not None
        assert ray_normal is not None
        assert ray_depth is not None
        assert ray_normal is not None

        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(self._data.quat_w_world[env_ids]).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * self._ray_directions_w[env_ids]),
                )
            )[:, :, 0]
            # apply the maximum distance after the transformation
            if self.cfg.depth_clipping_behavior == "max":
                distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
            elif self.cfg.depth_clipping_behavior == "zero":
                distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
                distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                ray_depth = torch.clip(ray_depth, max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                ray_depth[ray_depth > self.cfg.max_distance] = 0.0
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)
