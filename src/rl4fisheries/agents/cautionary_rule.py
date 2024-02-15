from itertools import product
import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm


class CautionaryRule:
    def __init__(self, x1=0, x2=1, y2=1, obs_bounds=1, **kwargs):
        from .unit_interface import unitInterface
        self.ui = unitInterface(bounds=obs_bounds)
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
        self.policy_type = "CautionaryRule_piecewise_linear"

        assert x1 <= x2, "CautionaryRule error: x1 <= x2" 

    def predict(self, state):
        pop = self.ui.to_natural_units(state)
        raw_prediction = np.clip( self.predict_raw(pop), 0, 1)
        return np.float32([2 * raw_prediction - 1])
    
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
    
    def save(self, path = None)->None:       
        path = path or os.path.join(f'{self.policy_type}.json')
        with open(path, 'w') as f:
            json.dump(
                self.__dict__, 
                f,
            )

    @classmethod
    def generate_tuning_stats(self, env, opt_msy, N=500, n_per_rad=40, n_rad=40, max_rad_frac=0.15, min_rad_frac=0.001):
        #
        from .evaluation import gather_stats
        #
        pbar = tqdm(
            product(
                np.linspace(min_rad_frac, max_rad_frac, n_rad), 
                range(n_per_rad)
            ), 
            desc="CautionaryRule.generate_tuning_stats()",
        )
        #
        results = []
        y2 = opt_msy
        for radius,  _ in pbar:
            theta = np.random.rand() * np.pi / 4
            x1 = np.sin(theta) * radius
            x2 = np.cos(theta) * radius
            # sin / cos are chosen so that x1 < x2
            manager = CautionaryRule(x1=x1, x2=x2, y2=y2)
            results.append([x1, x2, y2, *gather_stats(
                CautionaryRule(
                    x1=np.sin(theta) * radius, 
                    x2=np.cos(theta) * radius, 
                    y2=opt_msy,
                ), 
                env=env,
            )])

        return pl.from_records(results, schema=["x1",  "x2", "y2", "avg_rew", "low_rew", "hi_rew"])
        # def rand_pt(radius):
        #     theta = np.random.rand() * np.pi / 4
        #     return np.sin(theta) * radius, np.cos(theta) * radius

        # return pl.from_records(
        #     [
        #         [
        #             *rand_pt(radius),
        #             opt_msy,
        #             *gather_stats(
        #                 CautionaryRule(
        #                     x1=np.sin(theta) * radius, 
        #                     x2=np.cos(theta) * radius, 
        #                     y2=opt_msy
        #                 ), 
        #                 env=env,
        #             ),
        #         ] for radius, _ in pbar
        #     ],
        #     schema=["x1",  "x2", "y2", "avg_rew", "low_rew", "hi_rew"]
        # )
        

    
    @classmethod
    def load(self, path):
        assert path[-5:] == '.json', f"load error: msy.load(path) can only load json files currently!"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Target file is of type {type(data)}, it should be dict."

        # self.mortality = data.get("mortality")

        return CautionaryRule(**data)

    