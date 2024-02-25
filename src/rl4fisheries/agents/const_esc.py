import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm

class ConstEsc:
    def __init__(self, escapement=0, bounds = 1, **kwargs):
        from .unit_interface import unitInterface
        self.ui = unitInterface(bounds=bounds)
        self.escapement = escapement
        self.bounds = bounds
        self.policy_type = "constant_escapement"


    def predict(self, observation, **kwargs):
        pop = self.ui.to_natural_units(observation)
        raw_prediction = self.predict_raw(pop)
        return 2 * raw_prediction - 1, {}

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


        