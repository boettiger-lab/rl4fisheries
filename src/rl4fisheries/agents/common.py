import numpy as np
from typing import List, Text, Optional

def isVecObs(obs, env):
    if np.sum(np.shape(obs)) == 0:
        # scalar
        return False
    
    shp = env.observation_space.shape
    if (
        (shp != np.shape(obs)) and 
        (np.shape(obs[0]) == shp) # quick n dirty, possibly prone to bugs tho
    ):
        # deal with the possibility of a VecEnv observation
        return True
    #
    else:
        return False

def simulate_ep(env, agent, other_vars: Optional[List[Text]] = []):   
    simulation = {
        't': [],
        'obs': [],
        'act': [],
        'rew': [],
        **{var_name: [] for var_name in other_vars}
    }
    obs, _ = env.reset()
    for t in range(env.Tmax):
        action, _ = agent.predict(obs)
        new_obs, rew, term, trunc, info = env.step(action)
        #
        simulation['t'].append(t)
        simulation['obs'].append(obs)
        simulation['act'].append(act)
        simulation['rew'].append(rew)
        for var_name in other_vars:
            simulation[var_name].append(getattr(env, var_name))
        #
        obs = new_obs
    #
    return simulation
    