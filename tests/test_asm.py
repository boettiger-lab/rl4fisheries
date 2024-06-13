# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4fisheries import AsmEnvEsc, AsmEnv, AsmCRLike

def test_AsmEsc():
    check_env(AsmEnvEsc(), warn=True)

def test_AsmEnv():
    check_env(AsmEnv(), warn=True)

def test_AsmCRLike():
    check_env(AsmCRLike(), warn=True)
