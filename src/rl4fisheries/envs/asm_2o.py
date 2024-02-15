import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Optional

# equilibrium dist will in general depend on parameters, need a more robust way
# to reset to random but not unrealistic starting distribution
equib_init = [
    0.99999999,
    0.86000001,
    0.73960002,
    0.63605603,
    0.54700819,
    0.47042705,
    0.40456727,
    0.34792786,
    0.29921796,
    0.25732745,
    0.22130161,
    0.19031939,
    0.16367468,
    0.14076023,
    0.1210538,
    0.10410627,
    0.08953139,
    0.076997,
    0.06621742,
    0.40676419,
]


class Asm2o(gym.Env):
    """an age-structured model following the gym API standard"""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, render_mode: Optional[str] = 'rgb_array', config={}):
        config = config or {}
        parameters = {
            "n_age": 20,  # number of age classes
            "vbk": np.float32(0.23),  # von Bertalanffy kappa
            "s": np.float32(0.86),  # average survival
            "cr": np.float32(6.0),  # Goodyear compensation ratio
            "rinit": np.float32(0.01),  # initial number age-1 recruits
            "ro": np.float32(1.0),  # average unfished recruitment
            "uo": np.float32(0.12),  # average historical exploitation rate
            "asl": np.float32(0.5),  # vul par 1
            "ahv": np.float32(5.0),  # vul par 2
            "ahm": np.float32(6.0),  # age 50% maturity
            "upow": np.float32(1.0),  # 1 = max yield objective, < 1 = HARA
            "p_big": np.float32(0.05),  # probability of big year class
            "sdr": np.float32(0.3),  # recruit sd given stock-recruit relationship
            "rho": np.float32(0.0),  # autocorrelation in recruitment sequence
            "sdv": np.float32(1e-9),  # sd in vulnerable biomass (survey)
            "sigma": np.float32(1.5),
        }
        # these parameters can be specified in config
        self.n_year = config.get("n_year", 1000)
        self.Tmax = self.n_year
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.training = config.get("training", True)
        self.parameters = config.get("parameters", parameters)
        self.timestep = 0
        self.bound = 50  # a rescaling parameter
        self.parameters["ages"] = range(
            1, self.parameters["n_age"] + 1
        )  # vector of ages for calculations

        default_init = self.initialize_population()
        self.init_state = config.get("init_state", equib_init)
        
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.action_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        
        #self.reset()


    def step(self, action):
        mortality = self.mortality_units(action)
        self.state, reward = self.harvest(self.state, mortality)
        self.state = self.population_growth(self.state)
        self.timestep += 1
        terminated = bool(self.timestep >= self.n_year)

        # in training mode only: punish for population collapse
        # if sum(n) <= self.threshold: # note CB's code had this as well: `and self.training:`
        #    terminated = True
        #    reward -= 50/self.timestep

        observation = self.observe()
        return observation, np.float64(reward), terminated, False, {}
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 2
        scale = self.screen_width / world_width
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        total = self.population_units()
        y = 2 * total / self.screen_height - 1
        y = int(np.clip(y, [0], [self.screen_height]))
        x = int((self.n_year / self.Tmax ) * self.screen_width)
        y = x
        gfxdraw.filled_circle( # x, y, rad, color
                self.surf, x, y, int(4), (128, 128, 128)
            )
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def initialize_population(self):
        p = self.parameters  # snag those pars
        ninit = np.float32([0] * p["n_age"])  # initial numbers
        vul = ninit.copy()  # vulnerability
        wt = ninit.copy()  # weight
        mat = ninit.copy()  # maturity
        Lo = ninit.copy()  # survivorship unfished
        Lf = ninit.copy()  # survivorship fished
        mwt = ninit.copy()  # mature weight

        # leading array calculations to get vul-at-age, wt-at-age, etc.
        for a in range(0, p["n_age"], 1):
            vul[a] = 1 / (1 + np.exp(-p["asl"] * (p["ages"][a] - p["ahv"])))
            wt[a] = pow(
                (1 - np.exp(-p["vbk"] * p["ages"][a])), 3
            )  # 3 --> isometric growth
            mat[a] = 1 / (1 + np.exp(-p["asl"] * (p["ages"][a] - p["ahm"])))
            if a == 0:
                Lo[a] = 1
                Lf[a] = 1
            elif a > 0 and a < (p["n_age"] - 1):
                Lo[a] = Lo[a - 1] * p["s"]
                Lf[a] = Lf[a - 1] * p["s"] * (1 - vul[a - 1] * p["uo"])
            elif a == (p["n_age"] - 1):
                Lo[a] = Lo[a - 1] * p["s"] / (1 - p["s"])
                Lf[a] = (
                    Lf[a - 1]
                    * p["s"]
                    * (1 - vul[a - 1] * p["uo"])
                    / (1 - p["s"] * (1 - vul[a - 1] * p["uo"]))
                )
        
        ninit = np.array(p["rinit"]) * Lf
        mwt = mat * np.array(wt)
        sbro = sum(Lo * mwt)  # spawner biomass per recruit in the unfished condition
        bha = p["cr"] / sbro  # beverton-holt alpha
        bhb = (p["cr"] - 1) / (p["ro"] * sbro)  # beverton-holt beta

        # put it all in self so we can reference later
        self.parameters["Lo"] = Lo
        self.parameters["Lf"] = Lf
        self.parameters["vul"] = vul
        self.parameters["wt"] = wt
        self.parameters["min_wt"] = np.min(wt)
        self.parameters["max_wt"] = np.max(wt)
        self.parameters["mwt"] = mwt
        self.parameters["bha"] = bha
        self.parameters["bhb"] = bhb
        self.parameters["p_big"] = 0.05
        self.parameters["sdr"] = 0.3
        self.parameters["rho"] = 0
        n = np.array(ninit, dtype=np.float32)
        self.state = np.clip(n, 0, np.Inf)
        return self.state

    def harvest(self, n, mortality):
        p = self.parameters
        self.vulb = sum(p["vul"] * n * p["mwt"])
        self.vbobs = self.vulb  # could multiply this by random deviate
        self.ssb = sum(p["mwt"] * n)
        if sum(n) > 0:
            self.abar = sum(p["vul"] * np.array(p["ages"]) * n) / sum(n)
            self.wbar = sum(p["vul"] * n * p["wt"]) / sum(n * p["wt"])
        else:
            self.abar = 0
            self.wbar = 0
        self.yieldf = mortality[0] * self.vulb  # fishery yield
        reward = self.yieldf ** p["upow"]  # this is utility
        n = p["s"] * n * (1 - p["vul"] * mortality)  # eat fish
        return n, reward

    def population_growth(self, n):
        p = self.parameters
        # mu = np.log(1) - p["sigma"] ** 2 / 2
        bh_alpha = p["bha"]  # * np.random.lognormal(mu, p["sigma"])

        n[p["n_age"] - 1] = (
            n[p["n_age"] - 1] + n[p["n_age"] - 2]
        )  # plus group accounting
        for a in range(p["n_age"] - 2, 0, -1):
            n[a] = n[a - 1]  # advance fish one a

        n[0] = (
            (bh_alpha)
            * self.ssb
            / (1 + p["bhb"] * self.ssb)
            * self.r_devs[self.timestep]
        )  # NOTE eventually needs to be r_devs[t]
        return n

    def mean_wt_obs(self):
        self.state 
    
    def observe(self):
        total_pop = np.sum(self.state)
        pop_obs = 2 * total_pop / self.bound - 1

        mean_wt = np.sum(self.parameters["wt"] * self.state) / total_pop
        max_wt, min_wt = self.parameters["max_wt"], self.parameters["min_wt"] # for readability
        mean_wt_obs = (
            2 * (mean_wt - min_wt) / (max_wt - min_wt) - 1 
        )
        
        observation = np.clip(np.array([pop_obs, mean_wt_obs]), -1, 1)
        return np.float32(observation)

    def population_units(self):
        total = np.array([sum(self.state)])
        return total

    def mortality_units(self, action):
        action = np.clip(action, [-1], [1])
        mortality = (action + 1.0) / 2.0
        return mortality

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.initialize_population()
        self.state = self.init_state * np.array(
            np.random.uniform(0.1, 1), dtype=np.float32
        )
        self.r_devs = get_r_devs(
            n_year=self.n_year,
            p_big=self.parameters["p_big"],
            sdr=self.parameters["sdr"],
            rho=self.parameters["rho"],
        )
        obs = self.observe()
        return obs, {}


def get_r_devs(n_year, p_big=0.05, sdr=0.3, rho=0):
    """
    f(x) to create recruitment deviates, which are multiplied
    by the stock-recruitment prediction in the age-structured model

    args:
    n_year: number of deviates required for simulation
    p_big: Pr(big year class)
    r_big: magnitude of big year class
    sdr: sd of recruitment
    rho: autocorrelation in recruitment sequence
    returns:
    vector of recruitment deviates of length n_year

    """
    r_mult = np.float32([1] * n_year)
    u_rand = np.random.uniform(0, 1, n_year)
    n_rand = np.random.normal(0, 1, n_year)
    r_big = np.random.uniform(10, 30, n_year)

    r_low = (1 - p_big * r_big) / (1 - p_big)  # small rec event
    r_low = np.clip(r_low, 0, None)
    dev_last = 0
    for t in range(0, n_year, 1):
        r_mult[t] = r_low[t]
        if u_rand[t] < p_big:
            r_mult[t] = r_big[t]
        r_mult[t] = r_mult[t] * np.exp(sdr * n_rand[t] + rho * dev_last)
        dev_last = sdr * n_rand[t] + rho * dev_last
    return r_mult


# smoke-test
# Confirm environment is correctly defined:
# from stable_baselines3.common.env_checker import check_env
# check_env(asm(), warn=True)
