#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# gp
python fixed_policy_opt.py -p msy -v True -o gp &
python fixed_policy_opt.py -p esc -v True -o gp &
python fixed_policy_opt.py -p cr -v True -o gp &

# gbrt
python fixed_policy_opt.py -p msy -v True -o gbrt &
python fixed_policy_opt.py -p esc -v True -o gbrt &
python fixed_policy_opt.py -p cr -v True -o gbrt &