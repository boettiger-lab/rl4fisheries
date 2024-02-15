# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4fisheries import Asm, Asm2o, AsmEsc

def test_Asm():
    check_env(Asm(), warn=True)

def test_Asm2o():
    check_env(Asm2o(), warn=True)

def test_AsmEsc():
    check_env(AsmEsc(), warn=True)
