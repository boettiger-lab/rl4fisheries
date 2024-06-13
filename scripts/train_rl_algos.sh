#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# just for good measure, install
# pip install -e ..

# hf
python hf_login.py

# train
python train.py -f ../hyperpars/for_results/ppo_both_UM1.yml
python train.py -f ../hyperpars/for_results/ppo_mwt_UM1.yml
python train.py -f ../hyperpars/for_results/ppo_biomass_UM1.yml

python train.py -f ../hyperpars/for_results/ppo_both_UM2.yml
python train.py -f ../hyperpars/for_results/ppo_mwt_UM2.yml
python train.py -f ../hyperpars/for_results/ppo_biomass_UM2.yml

python train.py -f ../hyperpars/for_results/ppo_both_UM3.yml
python train.py -f ../hyperpars/for_results/ppo_mwt_UM3.yml
python train.py -f ../hyperpars/for_results/ppo_biomass_UM3.yml
