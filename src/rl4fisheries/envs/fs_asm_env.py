# frame stacked env
from rl4fisheries import AsmEnv
from gymnasium.wrappers import FrameStackObservation
import gymnasium as gym


class FrameStackedAsmEnv(gym.Env):
    def __init__(self, config={}):
        self.stack_size = config.get('stack_size', 4)
        self.base_env = AsmEnv(config = config.get('asm_config', {}))
        self.stacked_env = FrameStackObservation(
            self.base_env,
            stack_size = self.stack_size,
        )
        self.observation_space = self.stacked_env.observation_space
        self.action_space = self.stacked_env.action_space

    def reset(self, *, seed=None, options=None):
        return self.stacked_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.stacked_env.step(action)


