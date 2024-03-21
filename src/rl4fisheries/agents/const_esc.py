import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm

from rl4fisheries.agents.common import isVecObs

class ConstEsc:
    def __init__(self, env, escapement=0, bounds = 1, **kwargs):
        from .unit_interface import unitInterface
        self.ui = unitInterface(bounds=bounds)
        self.escapement = escapement
        self.bounds = bounds
        self.policy_type = "constant_escapement"
        self.env = env


    def predict(self, observation, **kwargs):
        if isVecObs(observation, self.env):
            observation = observation[0]
        pop = self.ui.to_natural_units(observation)
        raw_prediction = self.predict_raw(pop)
        return np.float32([2 * raw_prediction - 1]), {}

    def predict_raw(self, pop):
        population = pop[0]
        if population <= self.escapement or population == 0:
            return 0
        else:
            return (population - self.escapement) / population

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2

    def state_to_pop(self, state):
        return (state + 1 ) / 2


        