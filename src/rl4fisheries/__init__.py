# Importing from sub-directories here makes these available as 'top-level' imports
from rl4fisheries.envs.asm_env import AsmEnv
from rl4fisheries.envs.asm_esc import AsmEnvEsc
from rl4fisheries.envs.asm_cr_like import AsmCRLike
from rl4fisheries.envs.fs_asm_env import FrameStackedAsmEnv
from rl4fisheries.envs.asm_anger import AsmAngerEnv, FrameStackedAngerAsmEnv
from rl4fisheries.envs.asm_gas import AsmGas
from rl4fisheries.envs.asm_mv_avg import AsmMovingAvg

from rl4fisheries.agents.cautionary_rule import PrecautionaryPrinciple
from rl4fisheries.agents.const_esc import ConstantEscapement
from rl4fisheries.agents.msy import FMsy
from rl4fisheries.agents.const_act import ConstantAction


from gymnasium.envs.registration import register
# action is fishing intensity
register(id="AsmEnv", entry_point="rl4fisheries.envs.asm_env:AsmEnv")
# action is 'escapement'
register(id="AsmEnvEsc", entry_point="rl4fisheries.envs.asm_esc:AsmEnvEsc")
# CR-like actions
register(id="AsmCRLike", entry_point="rl4fisheries.envs.asm_cr_like:AsmCRLike")
# frame-stacked env
register(
    id="FrameStackedAsmEnv", 
    entry_point="rl4fisheries.envs.fs_asm_env:FrameStackedAsmEnv",
)
# env in which anger level increases b/c of small actions
register(
    id="AsmAngerEnv", 
    entry_point="rl4fisheries.envs.asm_anger:AsmAngerEnv",
)
# frame-stacked anger env
register(
    id="FrameStackedAngerAsmEnv", 
    entry_point="rl4fisheries.envs.asm_anger:FrameStackedAngerAsmEnv",
)
# gas price env
register(
    id="AsmGas", 
    entry_point="rl4fisheries.envs.asm_gas:AsmGas",
)
# moving avg env
register(
    id="AsmMovingAvg", 
    entry_point="rl4fisheries.envs.asm_mv_avg:AsmMovingAvg",
)


