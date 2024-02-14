#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
args = parser.parse_args()

from rl4fisheries import Asm, Asm2o

from rl4eco.utils import sb3_train    
sb3_train(args.file)