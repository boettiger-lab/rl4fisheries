from rl4fisheries import CautionaryRule, Msy, ConstEscapement, AsmEnv

def test_CR_biomass():
    env = AsmEnv()
    agent = CautionaryRule(env=env, x1=0, x2=env.bound, y2=1, observed_var='biomass')

    (
        assert (
            (agent.x1_pm1 == -1) and
            (agent.x2_pm1 == +1) and
            (agent.y2_pm1 == +1)
        ),
        "CR agent: Conversion of policy to [-1,+1] space lead to inconsistencies."
    )

    pred1, info1 = agent.predict(observation=-1)[0]
    pred2, info2 = agent.predict(observation=+1)

    (
        assert (
            (pred1 == -1) and
            (pred == +1) and
        ),
        "CR agent: agent.predict does note return expected prediction."
    )
    (
        assert (isinstance(info1, dict)) and (isinstance(info2, dict)),
        "CR agent: agent.predict returns a non-dict info."
    )

        