#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", choices = ["msy", "esc", "cr"], help="Policy to be tuned", type=str)
parser.add_argument("-v", "--verbose", help="Verbosity of tuning method", type=bool)
parser.add_argument("-o", "--opt-algo", choices=["gp", "gbrt"], help="Optimization algo used")
args = parser.parse_args()

from huggingface_hub import hf_hub_download, HfApi, login
import numpy as np
from skopt.space import Real
from skopt.utils import use_named_args
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rl4fisheries import AsmEnv

# hf login
# api = HfApi()
# login()

# optimization algo
if args.opt_algo == "gp":
    from skopt import gp_minimize
    opt_algo = gp_minimize
elif args.opt_algo == "gbrt":
    from skopt import gbrt_minimize
    opt_algo = gbrt_minimize

# policy
if args.policy == "msy":
    from rl4fisheries import Msy
    policy_cls = Msy
elif args.policy == "esc":
    from rl4fisheries import ConstEsc
    policy_cls = ConstEsc
elif args.policy == "cr":
    from rl4fisheries import CautionaryRule
    policy_cls = CautionaryRule


# optimizing space
msy_space = [Real(0.002, 0.25, name='mortality')]
esc_space = [Real(0.02, 0.15, name='escapement')]
cr_space  = [
    Real(0.00001, 1, name='radius'),
    Real(0.00001, np.pi/4.00001, name='theta'),
    Real(0, 0.2, name='y2')
]
space = {'msy':msy_space, 'esc':esc_space, 'cr':cr_space}[args.policy]

# optimizing function
from stable_baselines3.common.monitor import Monitor

@use_named_args(space)
def msy_fn(**params):
    agent = Msy(AsmEnv(), mortality=params['mortality'])
    env = AsmEnv()
    mean, sd = evaluate_policy(agent, Monitor(env), n_eval_episodes=100)
    return -mean

@use_named_args(space)
def esc_fn(**params):
    agent = ConstEsc(AsmEnv(), escapement=params['escapement'])
    env = AsmEnv()
    mean, sd = evaluate_policy(agent, Monitor(env), n_eval_episodes=100)
    return -mean

@use_named_args(space)
def cr_fn(**params):
    theta = params["theta"]
    radius = params["radius"]
    x1 = np.sin(theta) * radius
    x2 = np.cos(theta) * radius
    
    assert x1 <= x2, ("CautionaryRule error: x1 < x2, " + str(x1) + ", ", str(x2) )

    agent = CautionaryRule(AsmEnv(), x1 = x1, x2 =  x2, y2 = params["y2"])
    env = AsmEnv()
    mean, sd = evaluate_policy(agent, Monitor(env), n_eval_episodes=100)
    return -mean

opt_fn = {'msy':msy_fn, 'esc':esc_fn, 'cr':cr_fn}[args.policy]


# optimize
results = opt_algo(opt_fn, space, n_calls=300, verbose=args.verbose, n_jobs=-1)
print(
    f"{args.policy}-{args.opt_algo} results: "
    f"opt args = {[eval(f'{r:.4f}') for r in results.x]}, "
    f"rew={results.fun:.4f}"
)

# save
path = "../saved_agents/"
fname = f"{args.policy}_{args.opt_algo}.pkl"
dump(results, path+fname)

api.upload_file(
    path_or_fileobj=path+fname,
    path_in_repo="sb3/rl4fisheries/"+fname,
    repo_id="boettiger-lab/rl4eco",
    repo_type="model",
)


