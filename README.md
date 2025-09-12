# rl4fisheries

Open code to reproduce and extend the paper:

**Using machine learning to inform harvest control rule design in complex fishery settings**
Montealegre-Mora, Boettiger, Walters, Cahill.

We provide reinforcement learning (RL) and Bayesian optimization methodologies to optimize harvest control rules in several scenarios for a Walleye (_Sander vitreus_) population dynamics model.

## Install dependencies

To run the project, install [Calkit](https://github.com/calkit/calkit)
(which will install [`uv`](https://docs.astral.sh/uv) if not already
installed):

```sh
curl -LsSf https://github.com/calkit/calkit/raw/refs/heads/main/scripts/install.sh | sh
```

All other dependencies and environments will be handled automatically.

## Run the project to reproduce the results

After cloning the project, run the project pipeline with:

```sh
calkit run
```

This will train and tune models, then generate results and figures.
If you want to force rerunning stages with results cached, add the
`--force` or `-f` flag.

Note that because the system is stochastic,
new data and figures generated may not exactly match those in the paper.

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

This is done in the `train-rl-algo` pipeline stage defined in `calkit.yaml`.
It can be rerun with:

```sh
calkit run train-rl-algos
```

Similarly, to train all non-RL policies ('fixed policies' in the paper) in the 3 scenarios, run:

```sh
calkit run tune-fixed-policies
```

### Optimize policies in customized scenarios

To optimize an RL policy from scratch, use the command:

```sh
calkit xenv -n main -- python scripts/train.py -f path/to/config/file.yml
```

You can use the template config file `hyperpars/RL-template.yml` file to begin:

```bash
calkit xenv -n main -- python scripts/train.py -f hyperpars/RL-template.yml
```

For reference, the config files used for our paper's results are located at `hyperpars/for_results`.

Similarly, you can optimize fixed policies using:

```bash
calkit xenv -n main -- python scripts/tune.py -f path/to/config/file.yml,
```

e.g.:

```bash
calkit xenv -n main -- python scripts/tune.py -f hyperpars/for_results/fixed_policy_UM1.yml,
```

To add any of these to the pipeline, run the following command, replacing
stage name, input, output, and script paths as necessary:

```sh
calkit new python-script-stage \
    --name my-stage-name \
    --environment main \
    --script-path scripts/train.py \
    --arg -f --arg path/to/config/file.yml \
    --input src \
    --input path/to/config/file.yml \
    --output saved_agents/results/output-path-name.pkl
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
|-- calkit.yaml: Calkit config for defining environments and pipeline
|
|-- noxfile.py: file used to run continuous integration tests
|
|-- pyproject.toml: file used in the installation of this source code
|
|-- README.md
```
