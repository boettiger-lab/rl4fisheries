import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Optional

from rl4fisheries.envs.asm_fns import observe_1o, observe_2o, asm_pop_growth, harvest, render_asm, get_r_devs

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


class AsmEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = 'rgb_array', config={}):
        config = config or {}

        #
        # dynamical params (more added in self.initialize_population())
        self.parameters = {
            "n_age": 20,  # number of age classes
            "vbk": config.get("vbk" , np.float32(0.23)),  # von Bertalanffy kappa
            "s": config.get("s" , np.float32(0.86)),  # average survival
            "cr": config.get("cr" , np.float32(6.0)),  # Goodyear compensation ratio
            "rinit": config.get("rinit" , np.float32(0.01)),  # initial number age-1 recruits
            "ro": config.get("ro" , np.float32(1.0)),  # average unfished recruitment
            "uo": config.get("uo" , np.float32(0.12)),  # average historical exploitation rate
            "asl": config.get("asl" , np.float32(0.5)),  # vul par 1
            "ahv": config.get("ahv" , np.float32(5.0)),  # vul par 2
            "ahm": config.get("ahm" , np.float32(6.0)),  # age 50% maturity
            "upow": config.get("upow" , np.float32(1.0)),  # 1 = max yield objective, < 1 = HARA
            "p_big": config.get("p_big" , np.float32(0.05)),  # probability of big year class
            "sdr": config.get("sdr" , np.float32(0.3)),  # recruit sd given stock-recruit relationship
            "rho": config.get("rho" , np.float32(0.0)),  # autocorrelation in recruitment sequence
            "sdv": config.get("sdv" , np.float32(1e-9)),  # sd in vulnerable biomass (survey)
            "sigma": config.get("sigma" , np.float32(1.5)),
            # for survey_vul:
            "lbar": config.get("lbar", np.float32(57.57)),
            "linf": config.get("linf", np.float32(41.08)),
            "survey_phi": config.get("survey_phi", np.float32(2.02)),
        }
        self.parameters["ages"] = range(
            1, self.parameters["n_age"] + 1
        )  # vector of ages for calculations
        default_init = self.initialize_population()
        self.init_state = config.get("init_state", equib_init)
        
        #
        # episode params
        self.n_year = config.get("n_year", 1000)
        self.Tmax = self.n_year
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.timestep = 0
        self.bound = 50  # a rescaling parameter

        #
        # functions
        self._harvest_fn = config.get("harvest_fn", harvest)
        self._pop_growth_fn = config.get("pop_growth_fn", asm_pop_growth)
        self._render_fn = config.get("render_fn", render_asm)
        
        # _observation_fn defaults to observe_2o unless "observation_fn_id" or "observation_fn" specified
        obs_fn_choices = {"observe_1o": observe_1o, "observe_2o": observe_2o}        
        self._observation_fn = obs_fn_choices[
            config.get("observation_fn_id", "observe_2o")
        ]
        if "observation_fn" in config:
            self._observation_fn = config["observation_fn"]

        #
        # render params
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        #
        # gym API

        self.n_observs = config.get("n_observs", 2) # should match config["observation_fn"] or config["observation_fn_id"]!

        self.action_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array(self.n_observs * [-1], dtype=np.float32),
            np.array(self.n_observs * [1], dtype=np.float32),
            dtype=np.float32,
        )
    
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
        self.update_vuls()
        obs = self.observe()
        return obs, {}
    
    def step(self, action):
        # am i missing any steps? CHECK!
        #
        # update_vuls
        # update_ssb
        # harvest
        # update_vuls
        # update ssb
        # population_growth
        # observe
        
        self.update_vuls()
        self.update_ssb()
        #
        mortality = self.mortality_units(action)
        self.state, reward = self.harvest(mortality)
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

    """
    Customizable methods

    Note: is there a way to avoid the awkward self._fn(self) syntax?
    """
    def render(self):
        self._render_fn(self)
    
    def observe(self):
        return self._observation_fn(self)

    def harvest(self, mortality):
        return self._harvest_fn(self, mortality)

    def population_growth(self):
        return self._pop_growth_fn(self)

    """
    Fixed helper methods
    """
    def update_vuls(self):
        """ update vulnerable populations """
        self.surv_vul_pop = self.parameters["survey_vul"] * self.state
        self.harv_vul_pop = self.parameters["harvest_vul"] * self.state
        #
        self.surv_vul_n = sum(self.surv_vul_pop)
        self.harv_vul_n = sum(self.harv_vul_pop)
        #
        self.surv_vul_b = sum(self.surv_vul_pop * self.parameters["wt"])
        self.harv_vul_b = sum(self.harv_vul_pop * self.parameters["wt"])

    def update_ssb(self):
        self.ssb = sum(self.parameters["mwt"] * self.state)

    def mortality_units(self, action):
        action = np.clip(action, [-1], [1])
        mortality = (action + 1.0) / 2.0
        return mortality

    def population_units(self):
        return np.array([sum(self.state)])

    def initialize_population(self):
        p = self.parameters  # snag those pars
        ninit = np.float32([0] * p["n_age"])  # initial numbers
        survey_vul = ninit.copy()  # survey vulnerability
        harvest_vul = ninit.copy()  # harvest vulnerability
        wt = ninit.copy()  # weight
        mat = ninit.copy()  # maturity
        Lo = ninit.copy()  # survivorship unfished
        Lf = ninit.copy()  # survivorship fished
        mwt = ninit.copy()  # mature weight

        # leading array calculations to get vul-at-age, wt-at-age, etc.
        for a in range(0, p["n_age"], 1):
            survey_vul[a] = (p["linf"] / p["lbar"]) * (1 - np.exp(-p["vbk"] * p["ages"][a])) ** (p["survey_phi"])
            # 1 / (1 + np.exp(-p["asl"] * (p["ages"][a] - p["ahv"]))) # <- this harvest_vul was previously
            harvest_vul[a] = 1 / (1 + np.exp(-(p["ages"][a] - p["ahv"]) / p["asl"]))
            wt[a] = pow(
                (1 - np.exp(-p["vbk"] * p["ages"][a])), 3
            )  # 3 --> isometric growth
            mat[a] = 1 / (1 + np.exp(-p["asl"] * (p["ages"][a] - p["ahm"])))
            if a == 0:
                Lo[a] = 1
                Lf[a] = 1
            elif a > 0 and a < (p["n_age"] - 1):
                Lo[a] = Lo[a - 1] * p["s"]
                Lf[a] = Lf[a - 1] * p["s"] * (1 - harvest_vul[a - 1] * p["uo"])
            elif a == (p["n_age"] - 1):
                Lo[a] = Lo[a - 1] * p["s"] / (1 - p["s"])
                Lf[a] = (
                    Lf[a - 1]
                    * p["s"]
                    * (1 - harvest_vul[a - 1] * p["uo"])
                    / (1 - p["s"] * (1 - harvest_vul[a - 1] * p["uo"]))
                )
        
        ninit = np.array(p["rinit"]) * Lf
        mwt = mat * np.array(wt)
        sbro = sum(Lo * mwt)  # spawner biomass per recruit in the unfished condition
        bha = p["cr"] / sbro  # beverton-holt alpha
        bhb = (p["cr"] - 1) / (p["ro"] * sbro)  # beverton-holt beta

        # put it all in self so we can reference later
        self.parameters["Lo"] = Lo
        self.parameters["Lf"] = Lf
        self.parameters["survey_vul"] = survey_vul
        self.parameters["harvest_vul"] = harvest_vul # TBD: change!
        self.parameters["wt"] = wt
        self.parameters["min_wt"] = np.min(wt)
        self.parameters["max_wt"] = np.max(wt)
        self.parameters["mwt"] = mwt
        self.parameters["bha"] = bha
        self.parameters["bhb"] = bhb
        # self.parameters["p_big"] = 0.05 no need to reinitialize
        # self.parameters["sdr"] = 0.3
        # self.parameters["rho"] = 0
        n = np.array(ninit, dtype=np.float32)
        self.state = np.clip(n, 0, np.Inf)
        return self.state
