# rl4fisheries

Models:

- `asm_env.py`: provides `AsmEnv()`. This encodes our population dynamics model, coupled with an observation process, and a harvest process with a corresponding utility model. These processes can all be modified using the `config` argument. Their defaults are defined in `asm_fns.py`. By default, observations are stock biomass and mean weight.
- `asm_esc.py`: provides `AsmEscEnv()` which inherits from `AsmEnv` and has one difference to it: actions in `AsmEscEnv()` represent escapement levels rather than fishing intensities. 
- `ams_cr_like.py`: provides `AsmCRLike()`. In this environment, mean weight is observed and the action is to set parameters `(x1, x2, y2)` for a biomass-based harvest control rule of the type `CautionaryRule` (specified below).

Strategies evaluated with MSE: 

- `agents.cautionary_rule.CautionaryRule`: piece-wise linear harvest-control rule specified by three parameters `(x1, x2, y2)`. Example plot (TBD).
- `agents.msy.Msy`: constant mortality harvest control rule. Specified by one parameter `mortality`.
- `agents.const_esc.ConstEsc`: constant escapement harvest control rule. Specified by one parameter `escapement`.

## Installation

Clone this repo, then:

```bash
cd rl4fisheries
pip install .
```

## RL training:

Simply run 
```bash
python scripts/train.py -f path/to/config/file.yml
```
The trained model is automatically pushed to Huggingface (requires a HF token). 
The config files used for our results are found in `hyperpars/for_results/`

