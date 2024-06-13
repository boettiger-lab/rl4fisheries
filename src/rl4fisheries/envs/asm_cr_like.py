import gymnasium as gym
import numpy as np

from rl4fisheries import AsmEnv
from rl4fisheries.envs.asm_fns import observe_mwt

class AsmCRLike(AsmEnv):
    """ observe mean weight, decide on a CR-like policy for biomass. """
    def __init__(self, render_mode = 'rgb_array', config={}):
        super().__init__(render_mode=render_mode, config=config)

        self._observation_fn = observe_mwt

        self.action_space = gym.spaces.Box(
            np.array(2 * [-1], dtype=np.float32),
            np.array(2 * [1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.initialize_population()
        #
        # if self.use_custom_harv_vul:
        #     self.parameters["harvest_vul"] = self.custom_harv_vul
        # if self.use_custom_surv_vul:
        #     self.parameters["survey_vul"] = self.custom_surv_vul
        #
        self.state = self.init_state * np.array(
            np.random.uniform(0.1, 1), dtype=np.float32
        )
        #
        if self.noiseless:
            self.r_devs = np.ones(shape = self.n_year)
        elif self.reproducibility_mode:
            self.r_devs = self.fixed_r_devs  
        else:
            self.r_devs = self.get_r_devs(
                n_year=self.n_year,
                p_big=self.parameters["p_big"],
                sdr=self.parameters["sdr"],
                rho=self.parameters["rho"],
            )
        #
        self.update_vuls()
        obs = self.observe()
        return obs, {}

    def step(self, action):
        self.update_vuls()
        self.update_ssb()
        #
        x1, x2, y2 = self.unnormalize_action(action)
        self.state, reward = self.harvest(x1, x2, y2)
        #
        self.update_vuls()
        self.update_ssb()
        #
        self.state = self.population_growth()
        self.timestep += 1
        terminated = bool(self.timestep >= self.n_year)
        # 
        observation = self.observe()
        #
        return observation, reward, terminated, False, {}

    def unnormalize_action(self, action):
        # x1 = 10 * (action[0] + 1) / 2
        x1 = 0
        x2 = 10 * (action[0] + 1) / 2
        x2 = max(x2,x1)
        y2 = (action[1] + 1) / 2
        return np.float32([x1, x2,y2])

    def harvest(self, x1, x2, y2):
        if  self.surv_vul_b < x1:
            intensity = 0
        elif x1 <= self.surv_vul_b < x2:
            intensity = y2 * (self.surv_vul_b - x1) / (x2 - x1)
        else: # vul_b >= x2
            intensity = y2

        f_yield = self.harv_vul_b * intensity
        new_state = self.parameters["s"] * self.state * (1 - self.parameters["harvest_vul"] * intensity)
        reward = f_yield  **  self.parameters["upow"]

        return new_state, reward

    def observe(self):
        return observe_mwt(self)

        










            
    