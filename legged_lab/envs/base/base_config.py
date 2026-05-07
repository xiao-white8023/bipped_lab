import math
from dataclasses import MISSING
 
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab.sensors.ray_caster import RayCasterCfg

from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

import legged_lab.mdp as mdp
from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.sensors.lidar import LidarCfg

from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCameraCfg
from legged_lab.sensors.grouped_ray_caster import GroupedRayCasterCamera
@configclass
class RewardCfg:
    pass

@configclass
class Privileged_Info:
    enable_feet_Privileged:bool = False
    enable_root_height_Privileged:bool= False
    enable_feet_contact_force:bool = False

@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False   # 是否启用高度扫描传感器
    prim_body_name: str = MISSING      # 扫描器挂载的机器人主体名称（必填，MISSING表示必填项）
    resolution: float = 0.1            # 扫描分辨率（米/像素）
    size: tuple = (1.6, 1.0)           # 扫描范围（宽1.6m，长1.0m）
    debug_vis: bool = False            # 是否可视化扫描结果（调试用）
    drift_range: tuple = (0.0, 0.0)    # 扫描漂移范围（模拟传感器误差）

@configclass
class CameraCfg:
    add_camera:bool=False
    add_camera_noise:bool=False
    camera: GroupedRayCasterCameraCfg = GroupedRayCasterCameraCfg(
        prim_path=MISSING, 
        pattern_cfg=MISSING
    ) 
    depth_max:float = MISSING
    depth_update_interval:int=MISSING
    depth_crop:tuple=MISSING
    depth_history_frames:int=8

@configclass
class left_feet_ray_caster_cfg: 
    add_left_feet_ray_caster:bool=False
    left_feet_ray_caster: RayCasterCfg = RayCasterCfg(
        pattern_cfg=MISSING,
        mesh_prim_paths=MISSING
    )
@configclass  
class right_feet_ray_caster_cfg: 
    add_right_feet_ray_caster:bool=False
    right_feet_ray_caster: RayCasterCfg = RayCasterCfg(
        pattern_cfg=MISSING,
        mesh_prim_paths=MISSING
    )

@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0   # 单轮仿真最大时长（20秒）
    num_envs: int = 4096                 # 并行仿真的环境数量（Isaac Lab支持大规模并行）
    env_spacing: float = 2.5             # 环境间距（米，避免不同环境的机器人碰撞）
    robot: ArticulationCfg = MISSING     # 机器人关节配置（必填，定义机器人的关节/刚体属性）
    terrain_type: str = MISSING          # 地形类型（必填，如flat/rough/stairs等）
    terrain_generator: TerrainGeneratorCfg = None  # 地形生成器配置（None用默认）
    max_init_terrain_level: int = 5      # 初始地形难度等级上限
    height_scanner: HeightScannerCfg = HeightScannerCfg()  # 高度扫描器配置
    lidar: LidarCfg = LidarCfg()                           # 激光雷达配置
    depth_camera: TiledCameraCfg = TiledCameraCfg()        # 深度相机配置（Tiled是拼接相机）
    
    camera:CameraCfg = CameraCfg()
    left_feet_ray_caster: left_feet_ray_caster_cfg=left_feet_ray_caster_cfg()
    right_feet_ray_caster: right_feet_ray_caster_cfg=right_feet_ray_caster_cfg()
    privileged_info:Privileged_Info = Privileged_Info()

@configclass
class RobotCfg:
    actor_obs_history_length: int = 10              # 智能体（Actor）观测历史长度（时序观测）
    critic_obs_history_length: int = 10             # 评论家（Critic）观测历史长度（AC架构专用）
    depth_history_frames:int=8
    action_scale: float = 0.25                      # 动作缩放系数（将模型输出映射到关节指令范围）
    terminate_contacts_body_names: list = []        # 接触后终止仿真的部位（如躯干碰地则结束）
    feet_body_names: list = []                      # 机器人足部名称列表（用于检测足地接触）
    depth_max:float = MISSING
    depth_update_interval:int=MISSING
    depth_crop:tuple=MISSING
    limit_angle:float=MISSING


@configclass
class ObsScalesCfg:
    # # 各观测维度的缩放系数（统一量纲）
    lin_vel: float = 1.0                  # 线速度缩放
    ang_vel: float = 1.0                  # 角速度缩放
    projected_gravity: float = 1.0        # 投影重力缩放
    commands: float = 1.0                 # 指令缩放
    joint_pos: float = 1.0                # 关节位置缩放
    joint_vel: float = 1.0                # 关节速度缩放
    actions: float = 1.0                  # 动作缩放
    height_scan: float = 1.0              # 高度扫描缩放
    feet_pos:float=1.0
    feet_vel:float=1.0
    contact_force: float = 0.01

@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()  # 观测缩放配置
    clip_observations: float = 100.0           # 观测值裁剪上限（防止异常值）
    clip_actions: float = 100.0                # # 动作值裁剪上限
    height_scan_offset: float = 0.5              #  # 高度扫描基线偏移（调整观测基线）
 
 
@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)   # # x方向线速度（前进/后退）
    lin_vel_y: tuple = (-0.5, 0.5)   # y方向线速度（左右平移）
    ang_vel_z: tuple = (-1.0, 1.0)   # z轴角速度（转向）
    heading: tuple = (-math.pi, math.pi)  #  # 航向角（-π~π）
 
 
@configclass
class CommandsCfg():
    resampling_time_range: tuple = (10.0, 10.0)  # 指令重采样间隔（10秒更新一次）
    rel_standing_envs: float = 0.2               # 20%的环境让机器人保持站立
    rel_heading_envs: float = 1.0                # 100%的环境启用航向指令
    heading_command: bool = True                 # 是否启用航向指令
    heading_control_stiffness: float = 0.5       # 航向跟踪刚度（越大越跟紧指令
    debug_vis: bool = True                        # # 可视化指令（调试用）
    ranges: CommandRangesCfg = CommandRangesCfg()   # # 指令范围配置
 
 
@configclass
class NoiseScalesCfg:
    # # 各观测维度的噪声幅度（模拟真实传感器误差）
    lin_vel: float = 0.2  ## 线速度噪声 
    ang_vel: float = 0.2  # # 角速度噪声
    projected_gravity: float = 0.05  # # 投影重力噪声
    joint_pos: float = 0.01   # # 关节位置噪声
    joint_vel: float = 1.5    # # 关节速度噪声
    height_scan: float = 0   # # 高度扫描噪声
 
 
@configclass
class NoiseCfg:
    add_noise: bool = True   # # 是否给观测添加噪声
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()  # # 噪声幅度配置
 
 
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
 
 
@configclass
class ActionDelayCfg:
    enable: bool = False      # 是否启用动作延迟
    params: dict = {"max_delay": 5, "min_delay": 0}  # # 延迟步数（最大5步）
 
 
@configclass
class DomainRandCfg:
    add_EventCfg:bool = True
    events: EventCfg = EventCfg()    # 随机化事件配置
    action_delay: ActionDelayCfg = ActionDelayCfg()  # 动作延迟配置
 

@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15  # # PhysX GPU刚体补丁数（性能调优）
    use_gpu=True,            # 开启 GPU 物理
    gpu_max_rigid_contact_count=2**23, # 必须改大，否则 4096 环境会崩
    gpu_max_rigid_patch_count=2**23,   # 同上
    gpu_heap_capacity=2**26,           # 必须改大，显存预留
 
@configclass
class SimCfg:
    '''
    物理引擎每0.005s更新一次  更新4次后，神经网络再进行下一次的更新
    '''
    dt: float = 0.005  # # 物理仿真步长（0.005秒=200Hz）
    decimation: int = 4  # 控制频率抽取系数（200/4=50Hz控制频率）
    physx: PhysxCfg = PhysxCfg()  # PhysX物理引擎配置

