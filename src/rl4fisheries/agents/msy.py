import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm

from rl4fisheries.agents.common import isVecObs

class Msy:
    def __init__(self, env, mortality: float =0, threshold: float =0, **kwargs):
        self.mortality = mortality
        self.threshold = threshold
        self.policy_type = "msy_and_threshold"
        self.env = env

    def predict(self, observation, **kwargs):
        if isVecObs(observation, self.env):
            observation = observation[0]
        pop = self.state_to_pop(observation)
        raw_prediction = raw_prediction = np.clip( self.predict_raw(pop), 0, 1)
        return np.float32([2 * raw_prediction - 1]), {}
    
    def predict_raw(self, pop):
        population = pop[0]
        if population < self.threshold:
            return 0
        else:
            return self.mortality

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2
    
    def state_to_pop(self, state):
        return (state + 1 ) / 2


        