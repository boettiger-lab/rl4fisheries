## Identical to asm but with actions defined as escapement rather than as harvest


import numpy as np
import gymnasium as gym

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


class AsmEsc(gym.Env):
    """an age-structured model following the gym API standard"""

    def __init__(self, config=None):
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
            "sigma": config.get("sigma", 1.5),
        }
        # these parameters can be specified in config
        self.n_year = config.get("n_year", 1000)
        self.Tmax = self.n_year
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.training = config.get("training", True)
        self.parameters = config.get("parameters", parameters)
        self.timestep = 0
        self.bound = 50  # a rescaling parameter
        self.esc_bound = 10
        self.parameters["ages"] = range(
            1, self.parameters["n_age"] + 1
        )  # vector of ages for calculations
        default_init = self.initialize_population()
        self.init_state = config.get("init_state", equib_init)

        self.reset()
        self.action_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        action = np.clip(action, [-1], [1])
        observation = self.observe()
        escapement = (action + 1.0) * self.esc_bound / 2.0
        obs = (observation + 1.0) * self.bound / 2.0
        if obs > 0:
            mortality = 1.0 - escapement / obs
            mortality = mortality * (mortality > np.array([0.0]))
        else:
            mortality = np.array([0])
        self.state, reward = self.harvest(self.state, mortality)
        self.state = self.population_growth(self.state)
        self.timestep += 1
        terminated = bool(self.timestep > self.n_year)

        observation = self.observe()
        return observation, np.float64(reward), terminated, False, {}

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
        self.parameters["mwt"] = mwt
        self.parameters["bha"] = bha
        self.parameters["bhb"] = bhb
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
        mu = np.log(1) - p["sigma"] ** 2 / 2
        bh_alpha = p["bha"] * np.random.lognormal(mu, p["sigma"])

        n[p["n_age"] - 1] = (
            n[p["n_age"] - 1] + n[p["n_age"] - 2]
        )  # plus group accounting
        for a in range(p["n_age"] - 2, 0, -1):
            n[a] = n[a - 1]  # advance fish one a
        n[0] = (
            (bh_alpha) * self.ssb / (1 + p["bhb"] * self.ssb) * 1.0
        )  # NOTE eventually needs to be r_devs[t]
        return n

    def observe(self):
        total = np.array([sum(self.state)])
        observation = 2 * total / self.bound - 1
        observation = np.clip(observation, [-1.0], [1.0])
        return np.float32(observation)

    def population_units(self):
        total = np.array([sum(self.state)])
        return total

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.initialize_population()
        self.state = self.init_state * np.array(
            np.random.uniform(0.1, 1), dtype=np.float32
        )
        obs = self.observe()
        return obs, {}

