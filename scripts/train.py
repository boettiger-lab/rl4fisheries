#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
args = parser.parse_args()

from gymnasium.envs.registration import register

# action is 'harvest'
register(id="Asm-v0", entry_point="rl4fisheries.asm:Asm")
# action is 'escapement'
register(id="AsmEsc-v0", entry_point="rl4fisheries.asm_esc:AsmEsc")
# action is harvest, but observes both total count and mean biomass
register(id="Asm2o-v0", entry_point="rl4fisheries.asm_2o:Asm2o")


from rl4eco.utils import sb3_train    
sb3_train(args.file)
