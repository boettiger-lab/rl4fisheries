import gymnasium as gym
import numpy as np
from numbers import Number
from typing import Optional, Union
import warnings

class unitInterface:
    def __init__(
        self, 
        *,
        env: Optional[gym.Env] = None, 
        bounds: Optional[Union[np.ndarray, Number, list]] = None,
    ):
        if env and bounds:
            warnings.warn("`env` and `bound` provided to unitInterface(). `bound` value will override the obs_bound value of the env.")

        if not (env or bounds):
            raise Warning("unitInterface() initializer needs either an `env` or a `bound` argument.")
        
        self.env = env
        self.bounds = bounds

        self.ignore_env = False
        if bounds:
            self.ignore_env = True
        
        tmp_bounds = bounds or env.bound
        self.bound_used = self.preprocess(tmp_bounds)


    def to_natural_units(self, vec: np.ndarray):
        """ [-1, 1] space to [0, bounds] space. """
        assert isinstance(vec, np.ndarray), "unitInterface.to_natural_units() `vec` argument must be an np.ndarray!"
        return self.bound_used * (vec + 1 ) / 2

    def to_norm_units(self, vec: np.ndarray):
        """ [0, bounds] to [-1, 1] space. """
#        assert isinstance(vec, np.ndarray), "unitInterface.to_norm_units() `vec` argument must be an np.ndarray!"
        return vec / self.bound_used
        self.bound_used * (vec + 1 ) / 2

    def preprocess(self, bounds):
        if isinstance(bounds, Number):
            assert bounds > 0, "unitInteface() `bounds` cannot be zero!"
            return np.array([bounds])

        if isinstance(bounds, list):
            assert len(bounds) > 0, "unitInteface() `bounds` cannot be an empty list!"
            assert min(bounds) > 0, "unitInterface() `bounds` entries must be strictly positive."
            return np.array(bounds)

        if isinstance(bounds, np.ndarray):
            assert len(bounds) > 0, "unitInteface() `bounds` cannot be an empty array!"
            assert min(bounds) > 0, "unitInterface() `bounds` entries must be strictly positive."
            return bounds

        else:
            if self.ignore_env:
                raise Warning("unitInterface() `bounds` argument: `bounds` not in any of the allowed formats (list, number, np.ndarray)")
            else:
                raise Warning("unitInterface() `env` argument: `env.bound` not in any of the allowed formats (list, number, np.ndarray)")
            