import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm

class Msy:
    def __init__(self, mortality: float =0, threshold: float =0, env = None, **kwargs):
        self.mortality = mortality
        self.threshold = threshold
        self.policy_type = "msy_and_threshold"
        self.env = env

    def predict(self, state):
        pop = self.state_to_pop(state)
        raw_prediction = raw_prediction = np.clip( self.predict_raw(pop), 0, 1)
        return np.float32([2 * raw_prediction - 1])
    
    def predict_raw(self, pop):
        population = pop[0]
        if population < self.threshold:
            return 0
        else:
            return self.mortality

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
    def generate_tuning_stats(self, env, N=500, n_morts=100, max_mort=0.25):
        #
        from .evaluation import gather_stats
        #
        pbar = tqdm(np.linspace(0,  max_effort, n_efforts))
        pbar.set_descriptioon("Msy().generate_tuning_stats()")
        #
        return pl.from_records(
            [
                [m, *gather_stats(Msy(m), env=env)] for m in pbar
            ],
            schema=["mortality", "avg_rew", "low_rew", "hi_rew"]
        )
        
    
    @classmethod
    def load(self, path):
        assert path[-5:] == '.json', f"load error: msy.load(path) can only load json files currently!"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Target file is of type {type(data)}, it should be dict."

        # self.mortality = data.get("mortality")

        return Msy(**data)
        
        