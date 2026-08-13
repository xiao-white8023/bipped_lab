from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_rough_env import G1ROUGHEnv
from legged_lab.envs.g1.g1_rough_cfg import G129WALK_ROUGHAGENTENV,G129WALK_ROUGHENVCFG
from legged_lab.envs.g1.g1_film_cfg import G129MOE_FILMAGENTENV,G129MOE_FILMENVCFG
from legged_lab.envs.g1.g1_film_env import G1MOEFILMEnv
from legged_lab.envs.g1.RENet_cfg import G1RENETAGENTCFG,G1RENETENVCFG
from legged_lab.envs.g1.RENet_env import G1RENetEnv
from legged_lab.envs.g1.atten_cfg import AttenAGENTENV,AttenCFG
from legged_lab.envs.g1.atten_env import AttenEnv
from legged_lab.envs.g1.squat_stand_cfg import SaquatStandAGENTENV,SaquatStandENVCFG
from legged_lab.envs.g1.squat_stand_env import SquatStandEnv
from legged_lab.envs.g1.g1_recovery_cfg import G123RECOVERYAGENTCFG,G123RECOVERYENVCFG
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv

from legged_lab.utils.task_registry import task_registry

task_registry.register("g1_rough",G1ROUGHEnv,G129WALK_ROUGHENVCFG(),G129WALK_ROUGHAGENTENV())
task_registry.register("g1_film",G1MOEFILMEnv,G129MOE_FILMENVCFG(),G129MOE_FILMAGENTENV())
task_registry.register("g1_renet",G1RENetEnv,G1RENETENVCFG(),G1RENETAGENTCFG())
task_registry.register("g1_atten",AttenEnv,AttenCFG(),AttenAGENTENV())
task_registry.register("g1_squart",SquatStandEnv,SaquatStandENVCFG(),SaquatStandAGENTENV())
task_registry.register("g1_recovery",G1RecoveryEnv,G123RECOVERYENVCFG(),G123RECOVERYAGENTCFG())
