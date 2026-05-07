from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

from legged_lab.envs.g1.g1_noamp_cfg import G129_ROUGHAGENTENV,G129_ROUGHENVCFG
from legged_lab.envs.g1.g1_noamp_env import G1_CFGEnv
from legged_lab.envs.g1.g1_walk_env import G1Env
from legged_lab.envs.g1.g1_walk_cfg import G129WALK_FLATAGENTENV,G129WALK_FLATENVCFG
from legged_lab.envs.g1.g1_rough_env import G1ROUGHEnv
from legged_lab.envs.g1.g1_rough_cfg import G129WALK_ROUGHAGENTENV,G129WALK_ROUGHENVCFG

from legged_lab.utils.task_registry import task_registry


from legged_lab.envs.g1.g1_moe_cfg import G129WALK_MOEROUGHENVCFG,G129WALK_MOEROUGHAGENTENV
from legged_lab.envs.g1.g1_moe_env import G1MOEROUGHEnv


task_registry.register("g1_walk",G1Env,G129WALK_FLATENVCFG(),G129WALK_FLATAGENTENV())
task_registry.register("g1_rough",G1ROUGHEnv,G129WALK_ROUGHENVCFG(),G129WALK_ROUGHAGENTENV())

task_registry.register("g1_NOAMP",G1_CFGEnv,G129_ROUGHENVCFG(),G129_ROUGHAGENTENV())
task_registry.register("g1_moe",G1MOEROUGHEnv,G129WALK_MOEROUGHENVCFG(),G129WALK_MOEROUGHAGENTENV())


