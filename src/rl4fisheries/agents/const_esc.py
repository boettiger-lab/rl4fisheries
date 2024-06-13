import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm

from rl4fisheries.agents.common import isVecObs

class ConstEsc:
    def __init__(self, env, escapement, observed_var='biomass', **kwargs):
        self.escapement = escapement
        self.policy_type = "constant_escapement"
        self.env = env
        self.observed_var = observed_var


    def predict(self, observation, **kwargs):
        if isVecObs(observation, self.env):
            observation = observation[0]
            
        if self.observed_var == 'biomass':
            pop = self.env.bound * (observation[0] + 1) / 2
            predicted_effort = self.predict_effort(pop)
        if self.observed_var == 'mean_wt':
            MIN_WT = self.env.parameters['min_wt']
            MAX_WT = self.env.parameters['max_wt']
            mwt = (
                MIN_WT + (MAX_WT - MIN_WT) * (observation[1] + 1) / 2
            )
            predicted_effort = self.predict_effort(mwt)
        
        return np.float32([2 * predicted_effort - 1]), {}

    def predict_effort(self, obs_value):
        if obs_value <= self.escapement or obs_value == 0:
            return 0
        else:
            return (obs_value - self.escapement) / obs_value

    def state_to_pop(self, state):
        return (state + 1 ) / 2


        