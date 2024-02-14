# rl4fisheries

Models:

- asm.py: provides `Asm()`.  Observes 1 dimension (total N).  Action is harvest
- asm_2o.py: provides `Asm2o()`. Observes 2 dimensions: total N and mean biomass (wt).
- ams_esc.py: escapement `AsmEsc()` escapement-encoded.

Methods: 

(For both 1d and 2d observations)
- MSE piecewise linear rule (in mortality space) 
- Constant Mortality
- Constant escapement


## RL training:

requires `rl4eco` package. Simply run `scripts/train.py` pointing at the chosen configuration. The trained model is automatically pushed to Huggingface (requires a HF token).

```bash
python scripts/train.py -f hyperpars/ppo-asm2o-v0-1.yml
```
