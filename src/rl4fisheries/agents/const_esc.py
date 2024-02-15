import json
import os
import numpy as np


class ConstEsc:
    def __init__(self, escapement=0, obs_bounds = 1, **kwargs):
        from .unit_interface import unitInterface
        self.ui = unitInterface(bounds=obs_bounds)
        self.escapement = escapement
        self.obs_bound = obs_bounds
        self.policy_type = "constant_escapement"


    def predict(self, state):
        pop = self.ui.to_natural_units(state)
        raw_prediction = self.predict_raw(pop)
        return 2 * raw_prediction - 1

    def predict_raw(self, pop):
        population = pop[0]
        if population <= self.escapement or population == 0:
            return 0
        else:
            return (population - self.escapement) / population

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2
    
    def save(self, path = None)->None:       
        path = path or os.path.join(f'{self.policy_type}.json')
        with open(path, 'w') as f:
            json.dump(
                self.__dict__, 
                f,
            )

    def state_to_pop(self, state):
        return (state + 1 ) / 2
    
    @classmethod
    def load(self, path):
        assert path[-5:] == '.json', f"load error: msy.load(path) can only load json files currently!"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Target file is of type {type(data)}, it should be dict."

        # self.mortality = data.get("mortality")

        return ConstEsc(**data)
        
        