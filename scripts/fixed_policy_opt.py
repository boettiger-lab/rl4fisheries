#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", choices = ["msy", "esc", "cr"], help="Policy to be tuned", type=str)
parser.add_argument("-v", "--verbose", help="Verbosity of tuning method", type=bool)
parser.add_argument("-o", "--opt-algo", choices=["gp", "gbrt"], help="Optimization algo used")
parser.add_argument("-ncalls", "--n-calls", help="Number of objective function calls used by optimizing algo", type=int)
parser.add_argument("-f", "--config-file", help="yaml file with env config")
parser.add_argument("-id", "--id", help="Identifier string", default="")
args = parser.parse_args()

from huggingface_hub import hf_hub_download, HfApi, login

import numpy as np
import yaml

from skopt import dump
from skopt.space import Real
from skopt.utils import use_named_args

from rl4fisheries import AsmEnv
from rl4fisheries.utils import evaluate_agent

login()

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

# config
with open(args.config_file, "r") as stream:
    config_file = yaml.safe_load(stream)
    config = config_file["config"]


# optimizing space
msy_space = [Real(0.0001, 0.5, name='mortality')]
esc_space = [Real(0.0001, 10, name='escapement')]
cr_space  = [
    Real(0.00001, 10, name='radius'),
    Real(0.00001, np.pi/4.00001, name='theta'),
    Real(0, 0.8, name='y2')
]
space = {'msy':msy_space, 'esc':esc_space, 'cr':cr_space}[args.policy]

# optimizing function
@use_named_args(space)
def msy_fn(**params):
    agent = Msy(AsmEnv(config=config), mortality=params['mortality'])
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(n_eval_episodes=200)
    return -m_reward

@use_named_args(space)
def esc_fn(**params):
    agent = ConstEsc(AsmEnv(config=config), escapement=params['escapement'])
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(n_eval_episodes=200)
    return -m_reward

@use_named_args(space)
def cr_fn(**params):
    theta = params["theta"]
    radius = params["radius"]
    x1 = np.sin(theta) * radius
    x2 = np.cos(theta) * radius
    #
    agent = CautionaryRule(AsmEnv(config=config), x1 = x1, x2 =  x2, y2 = params["y2"])
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(n_eval_episodes=200)
    return -m_reward

opt_fn = {'msy':msy_fn, 'esc':esc_fn, 'cr':cr_fn}[args.policy]


# optimize
results = opt_algo(opt_fn, space, n_calls=args.n_calls, verbose=args.verbose, n_jobs=-1)
print(
    "\n\n"
    f"{args.policy}-{args.opt_algo} results: "
    f"opt args = {[eval(f'{r:.4f}') for r in results.x]}, "
    f"rew={results.fun:.4f}"
    "\n\n"
)

# save
path = "../saved_agents/"
save_id = "" if args.id == "" else f"_{args.id}"
fname = f"{args.policy}_{args.opt_algo}{save_id}.pkl"
dump(results, path+fname)

# hf
api = HfApi()
api.upload_file(
    path_or_fileobj=path+fname,
    path_in_repo="sb3/rl4fisheries/"+fname,
    repo_id="boettiger-lab/rl4eco",
    repo_type="model",
)


