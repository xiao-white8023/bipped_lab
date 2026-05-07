import numpy as np

import legged_lab.terrains.hf_terrain_cfg as terrain_gen_perlin
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

Flat_terrain=TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "Flat_terrain": terrain_gen.MeshPlaneTerrainCfg(proportion=0.9)
    },
)

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.6, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
        # "Inverted_pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.4,
        #     step_height_range=(0.14, 0.16),
        #     step_width=0.30,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_28": terrain_gen.MeshPyramidStairsTerrainCfg(                   
        #     proportion=0.4,
        #     step_height_range=(0.14, 0.16),
        #     step_width=0.30,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),

    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "Inverted_pyramid_stairs__28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.28,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "Inverted_pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.30,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "Inverted_pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.32,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),    

        # "pyramid_stairs_28": terrain_gen.MeshPyramidStairsTerrainCfg(                  
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.28,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_28": terrain_gen.MeshPyramidStairsTerrainCfg(                   
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.30,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_30": terrain_gen.MeshPyramidStairsTerrainCfg(                   
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.23),
        #     step_width=0.32,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
   
        #"wave": terrain_gen.HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=5.0),
        
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0        
            
        # ),

        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        # ),        
        # "high_platform": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        # ),

        # 坡
        "up_slope_terrain":terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0837,0.55),
                platform_width=3,
                inverted = False
        ),
        "down_slope_terrain":terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0837,0.55),
                platform_width=3,
                inverted = True
        ),

        # "star": terrain_gen.MeshStarTerrainCfg(
                #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.7, gap_width_range=(0.1, 0.6), platform_width=3.0
        )
    },
)

ROUGH_PERLIN_TERRAINS_CFG=ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen_perlin.PerlinPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,

            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),
        "pyramid_stairs_28": terrain_gen.MeshPyramidStairsTerrainCfg(                  
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_high": terrain_gen_perlin.PerlinPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),
        "pyramid_stairs_inv": terrain_gen_perlin.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),
        "pyramid_stairs_inv_high": terrain_gen_perlin.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),
        "boxes": terrain_gen_perlin.PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            num_obstacles=20,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 1.5),
            obstacle_height_range=(0.05, 0.45),
            platform_width=1.5,
            border_width=0.0,
            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),

        "hf_pyramid_slope_inv": terrain_gen_perlin.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.7),
            platform_width=1.5,
            border_width=1.0,
            perlin_cfg=terrain_gen_perlin.PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            )
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.1, gap_width_range=(0.1, 0.6), platform_width=3.0
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=5.0),
    },
)

flex_terrain_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "Inverted_pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_28": terrain_gen.MeshPyramidStairsTerrainCfg(                  
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 独木桥
        "star_terrain" : terrain_gen.MeshStarTerrainCfg(
            proportion=0.15,
            num_bars=3,
            bar_width_range=(0.15,0.30),
            bar_height_range=(10,20),
            platform_width=3
        ),

        #梅花桩
        "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.1,
            
            # 中心初始平台的宽度，让机器人有平稳的出生点
            platform_width=3.0,
            
            # 石块的最大高低差 (单位：米)
            # 随着难度提升，石块的高度差会从 0 逐渐逼近这个最大值 (例如 15厘米)
            stone_height_max=0.15,
            
            # 石块宽度的随机范围 (单位：米)
            # 在高难度区域，系统会倾向于生成靠近范围下限 (0.2米) 的小石块
            stone_width_range=(0.2, 0.6),
            
            # 石块之间缝隙/距离的随机范围 (单位：米)
            # 距离越大，机器人需要跨步或跳跃的幅度就越大
            stone_distance_range=(0.1, 0.4),
            
            # 缝隙的深度 (单位：米)
            # -10.0 表示一个深深的“悬崖”，一旦脚踩进去就会触发 contact_sensor 的 terminate 惩罚
            holes_depth=-10
        ),

        # 坡
        "up_slope_terrain":terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0837,0.55),
                platform_width=3,
                inverted = False
        ),
        "down_slope_terrain":terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                slope_range=(0.0837,0.55),
                platform_width=3,
                inverted = True
        )

        # "boxes_terrain": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.3,  # 这个地形在整个大地图中出现的概率 (20%)
            
        #     # 中心初始平台的宽度，让 G1 刚出生时有个平稳的地方站立
        #     platform_width=3.0, 
            
        #     # --- 关键：难度起点 (Level 0) ---
        #     # 刚开始训练时，给机器人很少的、很大块的、很矮的箱子
        #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=15,          # 只有15个箱子
        #         height=0.25,             # 箱子高度只有 5 厘米 (很容易跨过去)
        #         size=(0.6, 0.6),         # 箱子表面很大 (60cm x 60cm)，闭着眼都能踩准
        #         max_yx_angle=0.0,        # 箱子表面是完全水平的
        #     ),
            
        #     # --- 关键：难度终点 (Level Max) ---
        #     # 训练到最后，给机器人密集的、很小的、很高的箱子
        #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=45,          # 密密麻麻45个箱子
        #         height=0.25,             # 箱子高达 25 厘米 (需要把腿抬很高)
        #         size=(0.25, 0.25),       # 箱子表面极小 (25cm x 25cm)，逼迫 MHA 网络精准找落脚点
        #         max_yx_angle=0.0,       # 箱子表面还有最高 15 度的随机倾斜，踩上去容易打滑
        #         degrees=True
        #     ),
            
        #     # --- 整体地形的随机噪声 ---
        #     # 让这些箱子不是完美在同一水平面上，而是参差不齐的
        #     # 负值表示有些箱子会凹陷下去，正值表示凸起来
        #     abs_height_noise=(-0.05, 0.1), # 高低落差在 -5cm 到 +10cm 之间随机
        # )
        }
)