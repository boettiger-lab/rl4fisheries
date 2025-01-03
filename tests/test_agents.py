from rl4fisheries import PrecautionaryPrinciple, FMsy, ConstantEscapement, AsmEnv

import numpy as np

#
# Cautionary rule:

def test_PrecPrinc_biomass():
    env = AsmEnv()
    agent = PrecautionaryPrinciple(env=env, x1=0, x2=env.bound, y2=1, observed_var='biomass')

    assert (
            (agent.x1_pm1 == -1) and
            (agent.x2_pm1 == +1) and
            (agent.y2_pm1 == +1)
    ), "PrecautionaryPrinciple agent: Conversion of policy to [-1,+1] space lead to inconsistencies."

    pred1, info1 = agent.predict(observation=np.array([-1, 0]))
    pred2, info2 = agent.predict(observation=np.array([+1,0]))

    assert (
            (pred1 == -1) and
            (pred2 == +1)
    ), "PrecautionaryPrinciple agent: agent.predict does not return expected prediction."
    
    assert (isinstance(info1, dict)) and (isinstance(info2, dict)), (
        "PrecautionaryPrinciple agent: agent.predict returns a non-dict info."
    )

def test_PrecPrinc_mwt():
    env = AsmEnv()
    agent = PrecautionaryPrinciple(
        env=env, 
        x1=env.parameters["min_wt"], 
        x2=env.parameters["max_wt"], 
        y2=1, 
        observed_var='mean_wt',
    )

    assert (
            (agent.x1_pm1 == -1) and
            (agent.x2_pm1 == +1) and
            (agent.y2_pm1 == +1)
    ), "PrecautionaryPrinciple agent: Conversion of policy to [-1,+1] space lead to inconsistencies."

    pred1, info1 = agent.predict(observation=np.array([0, -1]))
    pred2, info2 = agent.predict(observation=np.array([0, +1]))

    assert (
            (pred1 == -1) and
            (pred2 == +1)
    ), "PrecautionaryPrinciple agent: agent.predict does not return expected prediction."
    
    assert (isinstance(info1, dict)) and (isinstance(info2, dict)), (
        "PrecautionaryPrinciple agent: agent.predict returns a non-dict info."
    )

        