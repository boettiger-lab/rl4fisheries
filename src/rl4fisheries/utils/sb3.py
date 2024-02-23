from sb3_contrib import TQC, ARS
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3

def load_sb3_agent(algo, env, weights, policy = "MlpPolicy"):
    ALGO = algorithm(algo)
    model = ALGO(policy, env)
    agent = model.load(weights)
    return agent

# dictionary look up as a function...
def algorithm(algo):
    algos = {
        "PPO": PPO,
        "ARS": ARS,
        "TQC": TQC,
        "A2C": A2C,
        "SAC": SAC,
        "DQN": DQN,
        "TD3": TD3,
        "ppo": PPO,
        "ars": ARS,
        "tqc": TQC,
        "a2c": A2C,
        "sac": SAC,
        "dqn": DQN,
        "td3": TD3,
    }
    return algos[algo]
