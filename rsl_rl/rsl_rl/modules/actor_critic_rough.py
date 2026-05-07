from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .cnn_mlp import CnnMlp as CnnMlpModule

from rsl_rl.utils import resolve_nn_activation
from typing import Optional

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        CnnMlp=None,
        **kwargs,
    ):

        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        # ==========================================
        # 1. 从 kwargs 中动态提取所有 PIE 辅助任务的配置参数
        # ==========================================
        self.single_proprio_dim = kwargs.pop("single_proprio_dim", 102)
        
        # 历史编码器参数
        his_encoder_dims = kwargs.pop("his_encoder_dims", [256, 128])

        self.his_latent_dim = kwargs.pop("his_latent_dim", 64)
        
        # 速度估计参数
        self.use_vel_estimation = kwargs.pop("use_vel_estimation", False)
        vel_hidden_dim = kwargs.pop("vel_hidden_dim", [32])
        vel_activation_str = kwargs.pop("vel_activation", "elu")
        vel_dim = kwargs.pop("vel_dim", 3)
        # 地形重建参数
        self.use_terrain_recon = kwargs.pop("use_terrain_recon", False)
        terrain_hidden_dim = kwargs.pop("terrain_hidden_dim", [256, 128])
        terrain_activation_str = kwargs.pop("terrain_activation", "elu")
        terrain_scan_dim = kwargs.pop("terrain_scan_dim", 187)

        # 提取网格裁剪配置
        terrain_recon_front_only = kwargs.pop("terrain_recon_front_only", False)
        terrain_recon_grid_cols = kwargs.pop("terrain_recon_grid_cols", 17)
        terrain_recon_grid_rows = kwargs.pop("terrain_recon_grid_rows", 11)
        terrain_recon_x_min = kwargs.pop("terrain_recon_x_min", 0.0)

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        activation = resolve_nn_activation(activation)

        self.has_cnn = CnnMlp is not None
        if self.has_cnn:
            # 实例化 CNN 模块
            self.cnn = CnnMlpModule(**CnnMlp)
            
            # 获取深度图的维度信息，计算展平后的长度
            self.cnn_channels = CnnMlp["input_channels"]
            self.cnn_h = CnnMlp["input_dim"][0]
            self.cnn_w = CnnMlp["input_dim"][1]
            self.depth_flat_dim = self.cnn_channels * self.cnn_h * self.cnn_w
            
            # 减去深度图维度，剩下的就是真实的本体观测维度
            self.proprio_actor_dim = num_actor_obs - self.depth_flat_dim  # 这个就是历史总的维度了
            self.proprio_critic_dim = num_critic_obs
            
            # --- 动态构建：历史编码器 ---
            self.history_encoder = self._build_mlp(
                input_dim=self.proprio_actor_dim,
                hidden_dims=his_encoder_dims,
                activation_fn=activation,
                output_dim=self.his_latent_dim
            )
            
            # --- 动态构建：速度估计器 ---
            if self.use_vel_estimation:
                vel_act_fn = resolve_nn_activation(vel_activation_str)
                self.vel_estimator = self._build_mlp(
                    input_dim=self.his_latent_dim,
                    hidden_dims=vel_hidden_dim,
                    activation_fn=vel_act_fn,
                    output_dim=vel_dim
                )
                
            # --- 动态构建：地形解码器 ---
            if self.use_terrain_recon:
                # 动态计算地形网格的索引和输出维度
                if terrain_recon_front_only:
                    resolution = 0.1
                    size_x = (terrain_recon_grid_cols - 1) * resolution
                    x_min_idx = max(0, int((terrain_recon_x_min + size_x / 2) / resolution))
                    front_indices = []
                    for y_idx in range(terrain_recon_grid_rows):
                        for x_idx in range(x_min_idx, terrain_recon_grid_cols):
                            front_indices.append(y_idx * terrain_recon_grid_cols + x_idx)
                    
                    self.terrain_output_dim = len(front_indices)
                    self.register_buffer('terrain_front_indices', torch.tensor(front_indices, dtype=torch.long))
                else:
                    self.terrain_output_dim = terrain_scan_dim
                    self.terrain_front_indices = None

                terrain_act_fn = resolve_nn_activation(terrain_activation_str)
                actor_input_dim = self.single_proprio_dim + self.his_latent_dim + self.cnn.output_dim
                
                # 使用动态计算出的 terrain_output_dim (比如 99 或 187)
                self.terrain_decoder = self._build_mlp(
                    input_dim=actor_input_dim,
                    hidden_dims=terrain_hidden_dim,
                    activation_fn=terrain_act_fn,
                    output_dim=self.terrain_output_dim
                )
            
            mlp_input_dim_a = self.single_proprio_dim + self.his_latent_dim + self.cnn.output_dim
            mlp_input_dim_c = self.proprio_critic_dim
        
        else:
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs
            self.proprio_actor_dim = num_actor_obs
            self.proprio_critic_dim = num_critic_obs
            self.cnn_channels = 1
            self.cnn_h = 1
            self.cnn_w = 1
            self.cnn = nn.Identity()
        
        self.actor = self._build_actor(mlp_input_dim_a,actor_hidden_dims,activation,num_actions)
        self.critic = self._build_critic(mlp_input_dim_c,critic_hidden_dims,activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # 缓存容器
        self._last_his_latent: Optional[torch.Tensor] = None
        self._last_actor_input: Optional[torch.Tensor] = None

    def _build_mlp(self, input_dim, hidden_dims, activation_fn, output_dim):
        """通用多层感知机(MLP)动态构建工具。支持 hidden_dims=[] 直接线性映射"""
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(activation_fn)
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        return nn.Sequential(*layers)


    def _build_actor(self,mlp_input_dim_a,actor_hidden_dims,activation,num_actions):
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        return nn.Sequential(*actor_layers)
        

    def _build_critic(self,mlp_input_dim_c,critic_hidden_dims,activation):
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        return  nn.Sequential(*critic_layers)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self.act_inference(observations)
    @torch.jit.unused
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @torch.jit.unused
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @torch.jit.unused
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _process_obs(self, obs, is_actor: bool = True):
        if not is_actor:
            return obs # Critic 直接返回供 PPO 算法截取 GT

        if not self.has_cnn:
            return obs
            
        # 1. 拆分本体历史和深度图
        proprio_history = obs[:, :self.proprio_actor_dim]
        depth_flat = obs[:, self.proprio_actor_dim:]        
        
        # 2. 提取当前单帧状态 (历史数据最后 single_proprio_dim 维)
        current_proprio = proprio_history[:, -self.single_proprio_dim:]

        # 3. 将历史数据送入历史编码器
        his_latent = self.history_encoder(proprio_history)
        
        # === 核心修改 1：屏蔽 JIT 编译器的追踪 ===
        if not torch.jit.is_scripting() and self.training: 
            self._last_his_latent = his_latent
        # ==========================================

        # 4. 提取视觉特征
        depth_img = depth_flat.view(-1, self.cnn_channels, self.cnn_h, self.cnn_w)
        depth_latent = self.cnn(depth_img)
        
        # 5. 拼接成 Actor 最终的特征向量 (102 + his_latent + cnn_output)
        actor_input = torch.cat([current_proprio, his_latent, depth_latent], dim=-1)
        
        # === 核心修改 2：屏蔽 JIT 编译器的追踪 ===
        if not torch.jit.is_scripting() and self.training:
            self._last_actor_input = actor_input  
        # ==========================================

        return actor_input

    def predict_velocity(self):
        """供 PPO 调用的辅助任务接口"""
        if self.use_vel_estimation and self._last_his_latent is not None:
            return self.vel_estimator(self._last_his_latent)
        return None

    def predict_terrain(self):
        """供 PPO 调用的辅助任务接口"""
        if self.use_terrain_recon and self._last_actor_input is not None:
            return self.terrain_decoder(self._last_actor_input)
        return None
    def update_distribution(self, observations):
        observations = self._process_obs(observations, is_actor=True)
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        observations = self._process_obs(observations, is_actor=True)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self._process_obs(critic_observations, is_actor=False) # 这个critic观测也是包含8*18*32的图片的，送入到）process_obs中。 要想把critic不包含图片的128维信息。需要在env中让8*18*32的图片从critic中删去，在修改__init__()的逻辑就行了
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
