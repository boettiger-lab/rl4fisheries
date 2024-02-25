# Importing from sub-directories here makes these available as 'top-level' imports
from rl4fisheries.envs.asm import Asm
from rl4fisheries.envs.asm_2o import Asm2o
from rl4fisheries.envs.asm_esc import AsmEsc

from rl4fisheries.agents.cautionary_rule import CautionaryRule
from rl4fisheries.agents.const_esc import ConstEsc
from rl4fisheries.agents.msy import Msy


from gymnasium.envs.registration import register
# action is 'harvest'
register(id="Asm-v0", entry_point="rl4fisheries.envs.asm:Asm")
# action is 'escapement'
register(id="AsmEsc-v0", entry_point="rl4fisheries.envs.asm_esc:AsmEsc")
# action is harvest, but observes both total count and mean biomass
register(id="Asm2o-v0", entry_point="rl4fisheries.envs.asm_2o:Asm2o")
