import numpy as np

def observe_1o(env):
    observation = 2 * np.array([env.surv_vul_b]) / env.bound - 1
    observation = np.clip(observation, -1.0, 1.0)
    return np.float32(observation)

def observe_2o(env):
    # biomass obs:
    biomass_obs = 2 * env.surv_vul_b / env.bound - 1

    # mean weight:
    if env.vul_pop_total==0:
        vulnuerable_mean_wt = 0
    else:
        vulnuerable_mean_wt = env.surv_vul_b / env.surv_vul_n

    # mean weight obs:
    max_wt, min_wt = env.parameters["max_wt"], env.parameters["min_wt"] # for readability
    mean_wt_obs = (
        2 * (vulnuerable_mean_wt - min_wt) / (max_wt - min_wt) - 1 
    )

    # gathering results:
    observation = np.clip(np.array([biomass_obs, mean_wt_obs]), -1, 1)
    return np.float32(observation)

def asm_pop_growth(env):
    n_age = env.parameters["n_age"]
    new_state = np.zeros(shape = n_age)
    #
    new_state[0] = (
        env.parameters["bha"]
        * env.ssb / (1 + env.parameters["bhb"] * env.ssb)
        * env.r_devs[env.timestep]
    )
    #
    new_state[1:n_age-1] = env.state[0:n_age-2].copy() # advance fish an age class
    #
    new_state[n_age-1] =  (
        env.state[n_age - 1].copy() + env.state[n_age - 2].copy()
    ) # oldest fish class 'piles up'
    #
    return new_state

def harvest(env, mortality):
    # self.vulb = sum(p["harvest_vul"] * n * p["wt"])
    # self.vbobs = self.vulb  # could multiply this by random deviate # now done in env.update_vuls()
    p = env.parameters
    # env.ssb = sum(p["mwt"] * env.state) # now done in env.update_ssb()
    
    # Side effect portion of fn (tbd: discuss - abar and wbar not otherwise used in env)
    #
    if sum(env.state) > 0:
        env.abar = (
            sum(p["harvest_vul"] * np.array(p["ages"]) * env.state) 
            / sum(env.state)
        )
        env.wbar = (
            sum(p["harvest_vul"] * p["wt"] * env.state) 
            / sum(env.state * p["wt"])
        )
    else:
        env.abar = 0
        env.wbar = 0
    #
    #
    yieldf = mortality[0] * env.harv_vul_b  # fishery yield
    reward = yieldf ** p["upow"]  # this is utility
    new_state = p["s"] * env.state * (1 - p["harvest_vul"] * mortality)  # remove fish
    return new_state, reward

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

def render_asm(env):
    if env.render_mode is None:
        assert env.spec is not None
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            "You can specify the render_mode at initialization, "
            f'e.g. gym.make("{env.spec.id}", render_mode="rgb_array")'
        )
        return
    
    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as e:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gymnasium[classic-control]`"
        ) from e
    
    if env.screen is None:
        pygame.init()
        if env.render_mode == "human":
            pygame.display.init()
            env.screen = pygame.display.set_mode(
                (env.screen_width, env.screen_height)
            )
        else:  # mode == "rgb_array":
            env.screen = pygame.Surface((env.screen_width, env.screen_height))
    if env.clock is None:
        env.clock = pygame.time.Clock()
    
    world_width = 2
    scale = env.screen_width / world_width
    env.surf = pygame.Surface((env.screen_width, env.screen_height))
    env.surf.fill((255, 255, 255))
    
    total = env.population_units()
    y = 2 * total / env.screen_height - 1
    y = int(np.clip(y, [0], [env.screen_height]))
    x = int((env.n_year / env.Tmax ) * env.screen_width)
    y = x
    gfxdraw.filled_circle( # x, y, rad, color
            env.surf, x, y, int(4), (128, 128, 128)
        )
    env.surf = pygame.transform.flip(env.surf, False, True)
    env.screen.blit(env.surf, (0, 0))
    if env.render_mode == "human":
        pygame.event.pump()
        env.clock.tick(env.metadata["render_fps"])
        pygame.display.flip()
    
    elif env.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.screen)), axes=(1, 0, 2)
        )




























