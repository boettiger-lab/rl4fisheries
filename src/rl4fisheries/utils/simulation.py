import numpy as np
import ray

class evaluate_agent:
    def __init__(self, *, agent, env=None, ray_remote=False):
        self.agent = agent
        self.env = env or agent.env
        self.simulator = get_simulator(ray_remote=ray_remote)
        self.ray_remote = ray_remote
    
    def evaluate(self, return_episode_rewards=False, n_eval_episodes=50):
        if self.ray_remote:
            rewards = ray.get([
                self.simulator.remote(env=self.env, agent=self.agent) 
                for _ in range(n_eval_episodes)
            ])
            if ray.is_initialized():
                ray.shutdown()
        else:
            rewards = [
                self.simulator(env=self.env, agent=self.agent) 
                for _ in range(n_eval_episodes)
            ]
        #
        if return_episode_rewards:
            return rewards
        else:
            return np.mean(rewards)
        
    

def get_simulator(ray_remote = False):
    if ray_remote:
        @ray.remote
        def simulator(env, agent):
            results = []
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action = agent.predict(observation, deterministic=True)[0]
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                if terminated or done:
                    break
            return episode_reward
        return simulator
    else:
        def simulator(env, agent):
            results = []
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                if terminated or done:
                    break
            return episode_reward
        return simulator

class simulator_old:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    def simulate(self, reps=10):
        results = []
        env = self.env
        agent = self.agent
        for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                if terminated or done:
                    break
            results.append(episode_reward)      
        return results
