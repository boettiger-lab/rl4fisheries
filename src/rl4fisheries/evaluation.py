import numpy as np
import ray

@ray.remote
def gen_ep_rew(manager, env):
    episode_reward = 0.0
    observation, _ = env.reset()
    for t in range(env.Tmax):
        action = manager.predict(observation)
        observation, reward, terminated, done, info = env.step(action)
        episode_reward += reward
        if terminated or done:
            break
    return episode_reward


def gather_stats(manager, env, N=500, main_stat="mean"):
    results = ray.get([gen_ep_rew.remote(manager, env) for _ in range(N)])
    y = np.mean(results)
    #
    sigma = np.std(results)
    ymin = y - sigma
    ymax = y + sigma
    #
    return y, ymin, ymax 