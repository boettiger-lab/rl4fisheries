from itertools import product
import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm
from .unit_interface import unitInterface


class CautionaryRule:
    def __init__(self, x1=0, x2=1, y2=1, obs_bounds=1, **kwargs):
        self.ui = unitInterface(bounds=obs_bounds)
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
        self.policy_type = "CautionaryRule_piecewise_linear"

        assert x1 <= x2, "CautionaryRule error: x1 <= x2" 

    def predict(self, observation, **kwargs):
        pop = self.ui.to_natural_units(observation)
        raw_prediction = np.clip( self.predict_raw(pop), 0, 1)
        return np.float32([2 * raw_prediction - 1]), {}
    
    def predict_raw(self, pop):
        population = pop[0]
        if population < self.x1:
            return 0
        elif self.x1 <= population < self.x2:
            return self.y2 * (population - self.x1) / (self.x2 - self.x1)
        else:
            return self.y2

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2
    
    def state_to_pop(self, state):
        return (state + 1 ) / 2
    
