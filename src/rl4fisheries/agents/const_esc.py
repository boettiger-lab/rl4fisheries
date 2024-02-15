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
    def generate_tuning_stats(self, env, N=500, n_escs=100, max_esc=0.25):
        #
        from rl4fisheries.evaluation import gather_stats
        #
        pbar = tqdm(np.linspace(0,  max_esc, n_escs), desc="ConstEsc.generate_tuning_stats()")
        #
        return pl.from_records(
            [
                [m, *gather_stats(ConstEsc(m), env=env)] for m in pbar
            ],
            schema=["escapement", "avg_rew", "low_rew", "hi_rew"]
        )
    
    @classmethod
    def load(self, path):
        assert path[-5:] == '.json', f"load error: msy.load(path) can only load json files currently!"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Target file is of type {type(data)}, it should be dict."

        # self.mortality = data.get("mortality")

        return ConstEsc(**data)
        
        