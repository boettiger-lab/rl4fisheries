from itertools import product
import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm
from .unit_interface import unitInterface

from rl4fisheries.agents.common import isVecObs

class CautionaryRule:
    def __init__(self, env, x1=0, x2=1, y2=1, observed_var='biomass', **kwargs):
        self.policy_type = f"CautionaryRule_{observed_var}"
        self.env = env
        self.observed_var = observed_var
        
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2

        self.x1_pm1 = self.convert_to_pm1(x1, var_type = self.observed_var) 
        self.x2_pm1 = self.convert_to_pm1(x2, var_type = self.observed_var)
        self.y2_pm1 = self.convert_to_pm1(y2, var_type = 'action')

        assert x1 <= x2, "CautionaryRule error: x1 <= x2" 

    def convert_to_pm1(self, X, var_type):
        if var_type == 'biomass':
            return 2 * X / self.env.bound - 1
        elif var_type == 'mean_wt':
            MAX_WT = self.env.parameters["max_wt"]
            MIN_WT = self.env.parameters["min_wt"]
            return 2 * (X - MIN_WT) / (MAX_WT - MIN_WT) - 1
        elif var_type == 'action':
            return 2 * X - 1
    
    def predict(self, observation, **kwargs):
        if isVecObs(observation, self.env):
            observation = observation[0]
        if self.observed_var == 'biomass':
            raw_prediction = np.clip(self.predict_raw(observation[0]), -1, 1)
        if self.observed_var == 'mean_wt':
            raw_prediction = np.clip(self.predict_raw(observation[1]), -1, 1)
        return np.float32([raw_prediction]), {}
    
    def predict_raw(self, observation):
        if observation < self.x1_pm1:
            return -1
        elif self.x1_pm1 <= observation < self.x2_pm1:
            prediction = (
                -1 
                + (
                    (self.y2_pm1 + 1) 
                    * (observation - self.x1_pm1) 
                    / (self.x2_pm1 - self.x1_pm1)
                ) # -1 + (y2 - - 1) * fraction
            )
            return prediction
        else:
            return self.y2_pm1

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2
    
    def state_to_pop(self, state):
        return (state + 1 ) / 2
    
