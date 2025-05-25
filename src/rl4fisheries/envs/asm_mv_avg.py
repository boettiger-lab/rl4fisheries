from rl4fisheries import AsmEnv

import gymnasium as gym
import numpy as np

class AsmMovingBmsAvg(AsmEnv):
    """ adds an average last N biomass obs """
    def __init__(self, config):
        super().__init__(config=config)
        self.base_asm = AsmEnv(config=config) # potentially unneeded?
        self.avg_win_size = config.get("avg_win_size", 10) 
        #
        self.n_observs = self.base_asm.n_observs + 1
        self.observation_space = gym.spaces.Box(
            np.array(self.n_observs * [-1], dtype=np.float32),
            np.array(self.n_observs * [1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs_stack = np.float32([
            self.base_asm.reset(seed=seed, options=options)[0]
            for _ in range(self.avg_win_size)
        ])
        self.bms_stack = obs_stack[:,0]
        
        avg_win_bms = np.mean(self.bms_stack)
        bms_obs = obs_stack[-1,0]
        mwt_obs = obs_stack[0,-1]
        obs = np.float32([bms_obs, mwt_obs, avg_win_bms])
        #
        return obs, {} # who needs info, yolo

    def step(self, action):
        base_obs, rew, term, trunc, info = self.base_asm.step(action)
        self.bms_stack = np.insert(
            arr = self.bms_stack,
            obj = 0, # index where inserted
            values = base_obs[0],
        )[:-1]
        obs = np.float32([*base_obs, np.mean(self.bms_stack)])
        #
        return obs, rew, term, trunc, info


class AsmMovingAvg(AsmEnv):
    """ adds an average last N biomass obs """
    def __init__(self, config):
        super().__init__(config=config)
        self.base_asm = AsmEnv(config=config) # potentially unneeded?
        self.avg_win_size = config.get("avg_win_size", 3) 
        #
        # self.n_observs = self.base_asm.n_observs + 1
        # self.observation_space = gym.spaces.Box(
        #     np.array(self.n_observs * [-1], dtype=np.float32),
        #     np.array(self.n_observs * [1], dtype=np.float32),
        #     dtype=np.float32,
        # )
        # self.action_space = gym.spaces.Box(
        #     np.array([-1], dtype=np.float32),
        #     np.array([1], dtype=np.float32),
        #     dtype=np.float32,
        # )

    def reset(self, *, seed=None, options=None):
        self.obs_stack = np.float32([
            self.base_asm.reset(seed=seed, options=options)[0]
            for _ in range(self.avg_win_size)
        ])
        obs = np.mean(
            self.obs_stack,
            axis=0, # so = average observation (row)
        )
        #
        return obs, {} # who needs info, yolo

    def step(self, action):
        base_obs, rew, term, trunc, info = self.base_asm.step(action)
        self.obs_stack = np.insert(
            arr = self.obs_stack, # target for insertion
            obj = [0], # location of insertion (brackets to not flatten the array)
            values = base_obs, # thing to be inserted
            axis = 0, # insert a row into array (not column)
        )[:-1] # lose the oldest observation
        
        obs = np.mean(
            self.obs_stack, 
            axis = 0, # take the col-wise mean (ie 'avg row vector')
        )
        #
        return obs, rew, term, trunc, info


        
        