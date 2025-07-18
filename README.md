# rl4fisheries

Open code to reproduce and extend the paper 

**Using machine learning to inform harvest control rule design in complex fishery settings**
Montealegre-Mora, Boettiger, Walters, Cahill.

We provide reinforcement learning (RL) and Bayesian optimization methodologies to optimize harvest control rules in several scenarios for a Walleye (*Sander vitreus*) population dynamics model.

## Quickstart: Reproducing paper figures

We provide notebooks to reproduce the figures in the paper in the folder `notebooks/for_results/`.
- These figures are generated with csv data that is stored in this [link](https://huggingface.co/boettiger-lab/rl4eco/tree/main/rl4fisheries-reproducing).
- First step for reproducing results here is a script `setup.sh` which sets the python environment that the notebooks need in order to generate the figures.
- Notebook `2-download-data.ipynb` downloads the data locally to the user's computer. 
- Subsequent notebooks use that downloaded data to generate figures.

## Installation

To install this source code, you need to have git, Python and pip installed.
To quickly check whether these are installed you can open the terminal and run the following command:

```bash
pip install git+https://github.com/boettiger-lab/rl4fisheries.git
```

## Optimized policies

The optimized policies presented in the paper are saved at this [link](https://huggingface.co/boettiger-lab/rl4eco/tree/main/sb3/rl4fisheries/post-review-results/) which is a public repository hosted by [Hugging Face](https://huggingface.co).
That link contains a variety of policies which we trained using different algorithms, neural network architectures and fishery scenarios.
The policies used in our paper may be found on [this notebook](https://github.com/boettiger-lab/rl4fisheries/blob/new-fig/notebooks/for_generating_results/2_reward_distr.ipynb), on cells 4 and 6:
- `cr-UM{1,2,3}-noise01.pkl` --> optimized precautionary policies
- `msy-UM{1,2,3}-noise01.pkl` --> constant finite exploitation rate ('const-U') policies
- `2obs-UM1-256-64-16-noise0.1-chkpnt2.zip` --> 2-observation RL policy for UM1
- `biomass-UM1-64-32-16-noise0.1-chkpnt2.zip` --> 1-observation RL policy for UM1
- `2obs-UM2-256-64-16-noise0.1-chkpnt4.zip` --> 2-observation RL policy for UM2
- `biomass-UM2-64-32-16-noise0.1-chkpnt4.zip` --> 1-observation RL policy for UM2
- `2obs-UM3-256-64-16-noise0.1-chkpnt2.zip` --> 2-observation RL policy for UM3
- `biomass-UM3-64-32-16-noise0.1-chkpnt4.zip` --> 1-observation RL policy for UM3

Above, we use the notation UM{1,2,3} to mean the three utility models examined in the paper: UM1 = harvest utility, UM2 = HARA utility, UM3 = trophy utility.

## Optimizing policies

Here we explain how to use our code to optimize policies as done in our paper.

### Train all RL/non-RL policies in all scenarios considered in paper

To train both RL policies in the 3 utility model scenarios considered in the paper, you can use the following command:
``` bash
bash scripts/train_RL_algos.sh
```

Similarly, to train all non-RL policies ('fixed policies' in the paper) in the 3 scenarios, run 
``` bash
bash scripts/tune_fixed_policies.sh
```

### Optimize policies in customized scenarios

To optimize an RL policy from scratch, use the command
```bash
python scripts/train.py -f path/to/config/file.yml
```
You can use the template config file `hyperpars/RL-template.yml` file to begin:
```bash
python scripts/train.py -f hyperpars/RL-template.yml
```
For reference, the config files used for our paper's results are located at `hyperpars/for_results`.

Similarly, you can optimize fixed policies using
``` bash
python scripts/tune.py -f path/to/config/file.yml,
```
e.g. 
``` bash
python scripts/tune.py -f hyperpars/for_results/fixed_policy_UM1.yml,
```

## Source code structure

```
rl4fisheries
|
|-- hyperpars
|   |
|   |-- configuration yaml files used by optimizing scripts
|
|-- notebooks
|   |
|   |-- Jupyter notebooks to reproduce results
|
|-- src/rl4fisheries
|   |
|   |-- agents
|   |   |
|   |   |-- interfaces for policies such as Precautionary Policy, UMSY, Constant Escapement
|   |
|   |-- envs
|   |   |
|   |   |-- Gymnasium environments used to model Walleye population dynamics in our study.
|   |       (specifically, asm_env.py is used for our paper).
|   |
|   |-- utils
|       |
|       |-- ray.py: RL training within Ray framework (not used in paper, potentially not working)
|       |
|       |-- sb3.py: RL training within Stable Baselines framework (used in paper)
|       |
|       |-- simulation.py: helper functions to simulate the system dynamics using a policy (used to generate policy evaluations)
|    
|-- tests
|   |
|   |-- continuous integration tests to ensure code quality in pull requests
|
|-- noxfile.py: file used to run continuous integration tests
|
|-- pyproject.toml: file used in the installation of this source code
|
|-- README.md 
```
