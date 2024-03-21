import numpy as np

def isVecObs(obs, env):
    shp = env.observation_space.shape
    if (
        (shp != np.shape(obs)) and 
        (np.shape(obs[0]) == shp) # quick n dirty, possibly prone to bugs tho
    ):
        return True
    return False