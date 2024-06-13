# Importing from sub-directories here makes these available as 'top-level' imports
from rl4fisheries.envs.asm import Asm
from rl4fisheries.envs.asm_2o import Asm2o
from rl4fisheries.envs.asm_env import AsmEnv

from rl4fisheries.envs.asm_esc import AsmEnvEsc
from rl4fisheries.envs.asm_cr_like import AsmCRLike

from rl4fisheries.agents.cautionary_rule import CautionaryRule
from rl4fisheries.agents.const_esc import ConstEsc
from rl4fisheries.agents.msy import Msy
from rl4fisheries.agents.const_act import ConstAct


from gymnasium.envs.registration import register
# action is 'harvest'
register(id="Asm-v0", entry_point="rl4fisheries.envs.asm:Asm")
# action is 'escapement'
register(id="AsmEnvEsc", entry_point="rl4fisheries.envs.asm_esc:AsmEnvEsc")
# action is harvest, but observes both total count and mean biomass
register(id="Asm2o-v0", entry_point="rl4fisheries.envs.asm_2o:Asm2o")
# action is harvest, but observes both total count and mean biomass
register(id="AsmEnv", entry_point="rl4fisheries.envs.asm_env:AsmEnv")
# CR-like actions
register(id="AsmCRLike", entry_point="rl4fisheries.envs.asm_cr_like:AsmCRLike")

