#!/opt/venv/bin/python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input-file", help="input yaml file")
parser.add_argument("-hf", "--do-huggingface-login", default=False, type=bool, help="Whether to log in to hugging face from tune.py script.")
parser.add_argument("-vb", "--opt-verbose", default=False, type=bool, help="GP optimizer verbosity.")
args = parser.parse_args()

import gymnasium as gym
import numpy as np
import yaml

from huggingface_hub import hf_hub_download, HfApi, login
from skopt import gp_minimize, dump
from skopt.space import Real
from skopt.utils import use_named_args

from rl4fisheries import AsmEnv, FMsy, ConstantEscapement, PrecautionaryPrinciple
from rl4fisheries.utils import evaluate_agent

print(f"""

Working with the input {args.input_file} now...

""")

with open(args.input_file, "r") as stream:
    OPTIONS = yaml.safe_load(stream)

"""
OPTIONS = {
    'config': {...}
    'n_eval_episodes': ...
    'n_calls': ...
    'id': ...
    'repo_id':
    'env_id': [optional]
}
"""

ENV_ID = OPTIONS.get("env_id", "AsmEnv")

#
#
### HF

if args.do_huggingface_login:
    login()
api = HfApi()

#
#
### OPTIMIZATION OBJECTIVE FNS

msy_space = [Real(0.0001, 0.5, name='mortality')]
esc_space = [Real(0.0001, 10, name='escapement')]
cr_space  = [
    Real(0.00001, 10, name='radius'),
    Real(0.00001, np.pi/4.00001, name='theta'),
    Real(0, 0.8, name='y2')
]

@use_named_args(msy_space)
def msy_fn(**params):
    # this can't possibly be 'the way'
    env_used = gym.make(ENV_ID, config=OPTIONS['config']).env.env
    agent = FMsy(
        env_used, 
        mortality=params['mortality'],
    )
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(
        n_eval_episodes=OPTIONS['n_eval_episodes']
    )
    return -m_reward

@use_named_args(esc_space)
def esc_fn(**params):
    env_used = gym.make(ENV_ID, config=OPTIONS['config']).env.env
    agent = ConstantEscapement(
        env_used, 
        escapement=params['escapement'],
    )
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(
        n_eval_episodes=OPTIONS['n_eval_episodes']
    )
    return -m_reward

@use_named_args(cr_space)
def cr_fn(**params):
    theta = params["theta"]
    radius = params["radius"]
    x1 = np.sin(theta) * radius
    x2 = np.cos(theta) * radius
    #
    env_used = gym.make(ENV_ID, config=OPTIONS['config']).env.env
    agent = PrecautionaryPrinciple(
        env_used, 
        x1 = x1, 
        x2 =  x2, 
        y2 = params["y2"],
    )
    m_reward = evaluate_agent(agent=agent, ray_remote=True).evaluate(
        n_eval_episodes=OPTIONS['n_eval_episodes']
    )
    return -m_reward

#
#
### OPTIMIZE

msy_results = gp_minimize(msy_fn, msy_space, n_calls=OPTIONS['n_calls'], verbose=args.opt_verbose)

print(msy_results.x, msy_results.fun)

# esc_results = gp_minimize(esc_fn, esc_space, n_calls=OPTIONS['n_calls'], verbose=args.opt_verbose)

# print(esc_results.x, esc_results.fun)

cr_results = gp_minimize(cr_fn, cr_space, n_calls=OPTIONS['n_calls'], verbose=args.opt_verbose)

print(cr_results.x, cr_results.fun)

print("All results:", end="\n-----------\n\n")

print(
    "\n\n"
    f"gp-msy results: "
    f"opt args = {[eval(f'{r:.4f}') for r in msy_results.x]}, "
    f"rew={msy_results.fun:.4f}"
    "\n\n"
)

print(
    "\n\n"
    f"gp-cr results: "
    f"opt args = {[eval(f'{r:.4f}') for r in cr_results.x]}, "
    f"rew={cr_results.fun:.4f}"
    "\n\n"
)

# print(
#     "\n\n"
#     f"gp-esc results: "
#     f"opt args = {[eval(f'{r:.4f}') for r in esc_results.x]}, "
#     f"rew={esc_results.fun:.4f}"
#     "\n\n"
# )

#
#
### SAVE

if True:

    path = "../saved_agents/results/"
    msy_fname = f"msy-{OPTIONS['id']}.pkl"
    # esc_fname = f"esc-{OPTIONS['id']}.pkl"
    cr_fname = f"cr-{OPTIONS['id']}.pkl"
    
    with open(path+msy_fname, "wb"):
        dump(res=msy_results, filename=path+msy_fname, store_objective=False)
    
    # with open(path+esc_fname, "wb"):
    #     dump(res=esc_results, filename=path+esc_fname, store_objective=False)
    
    with open(path+cr_fname, "wb"):
        dump(res=cr_results, filename=path+cr_fname, store_objective=False)

# HF

if True:
    
    api.upload_file(
        path_or_fileobj=path+msy_fname,
        path_in_repo="sb3/rl4fisheries/post-review-results/"+msy_fname,
        repo_id=OPTIONS["repo_id"],
        repo_type="model",
    )
    
    # api.upload_file(
    #     path_or_fileobj=path+esc_fname,
    #     path_in_repo="sb3/rl4fisheries/results/"+esc_fname,
    #     repo_id=OPTIONS["repo_id"],
    #     repo_type="model",
    # )
    
    api.upload_file(
        path_or_fileobj=path+cr_fname,
        path_in_repo="sb3/rl4fisheries/post-review-results/"+cr_fname,
        repo_id=OPTIONS["repo_id"],
        repo_type="model",
    )
