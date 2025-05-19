#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# hf
python hf_login.py

# gp
# python tune.py -f ../hyperpars/for_results/fixed_policy_UM1.yml
# python tune.py -f ../hyperpars/for_results/fixed_policy_UM2.yml
# python tune.py -f ../hyperpars/for_results/fixed_policy_UM3.yml

python tune.py -f ../hyperpars/noisy_obs/fixed_policy_UM1.yml 
python tune.py -f ../hyperpars/noisy_obs/fixed_policy_UM2.yml 
python tune.py -f ../hyperpars/noisy_obs/fixed_policy_UM3.yml 
