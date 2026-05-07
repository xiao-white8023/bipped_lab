from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation
from .actor_critic_rough import ActorCritic
from .moe import MoeLayer




class MoeActorCritic(ActorCritic):
    def __init__(self,
                num_actor_obs,
                num_critic_obs,
                num_actions,
                actor_hidden_dims,
                critic_hidden_dims,
                activation,
                init_noise_std=1.0,
                noise_std_type:str="scalar",
                num_experts=4,
                gate_hidden_dim=[],
                CnnMlp=None,
                **kwargs
                ):
        
        self.num_experts=num_experts
        self.gate_hidden_dim=gate_hidden_dim
        super().__init__(        
                num_actor_obs=num_actor_obs,
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                CnnMlp=CnnMlp, # <--- 传给基类
                **kwargs)
        
    def _build_actor(self,num_actor_obs,actor_hidden_dims,activation,num_actions):
        moe = MoeLayer(input_dim=num_actor_obs,
                     num_experts=self.num_experts,
                     output_dim=num_actions,
                     activation=activation,
                     expert_hidden_dims=actor_hidden_dims,
                     gate_hidden_dims=self.gate_hidden_dim
                    )
        return moe
    
    def _build_critic(self,num_critic_obs,critic_hidden_dims,activation):
        return MoeLayer(input_dim=num_critic_obs,
                        num_experts=self.num_experts,
                        output_dim=1,
                        activation=activation,
                        expert_hidden_dims=critic_hidden_dims,
                        gate_hidden_dims=self.gate_hidden_dim
                     )

