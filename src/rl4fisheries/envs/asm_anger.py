from rl4fisheries import AsmEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import numpy as np

class AsmAngerEnv(AsmEnv):
    def __init__(self, config={}, anger_config={}):
        super().__init__(config=config)
        self.anger_lvl = 1
        self.anger_config = anger_config
        self.action_threshold = anger_config.get('action_threshold', 0.01)

        self.n_observs += 1 # include action into observation for lstm
        self.observation_space = gym.spaces.Box(
            np.array(self.n_observs * [-1], dtype=np.float32),
            np.array(self.n_observs * [1], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self.anger_lvl = 1
        obs, info = super().reset(seed=seed, options=options)
        return np.append(obs, np.float32(-1.)), info
        

    def step(self, action):
        #
        # update anger
        if action < self.action_threshold: 
            self.anger_lvl += 1
        else:
            self.anger_lvl = max(self.anger_lvl-2, 1) # anger wantes twice as fast as it grows?

        obs, rew, term, trunc, info = super().step(action)
        rew = rew / (self.anger_lvl) ** 0.1
        obs = np.append(obs, np.float32(action))
        
        return obs, rew, term, trunc, info
        

class FrameStackedAngerAsmEnv(gym.Env):
    def __init__(self, config={}):
        self.stack_size = config.get('stack_size', 4)
        self.base_env = AsmAngerEnv(config = config.get('asm_anger_config', {}))
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