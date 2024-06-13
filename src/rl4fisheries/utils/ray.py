import gymnasium as gym
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from rl4fisheries import AsmEnv

def algorithm(algo_name, algo_config, env_name):
    algo_configs = {'PPO': PPOConfig, 'ppo': PPOConfig}
    return (
        algo_configs[algo_name]
        .training(algo_config)
        .build(env = env_name)
    )


def ray_train(config_file, **kwargs):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
        options = {**options, **kwargs}
    
    register_env(options["env_id"], lambda config: gym.make(options["env_id"], kwargs=options["config"]))

    # getting the config translated to Ray
    algo_config_sb3 = options["algo_config"]
    algo_config = {}
    if "learning_rate" in algo_config_sb3:
        algo_config["lr"] = algo_config_sb3["learning_rate"]
    if "labmda" in algo_config_sb3:
        algo_config["lambda"] = algo_config_sb3["lambda"]
    if "tau" in algo_config_sb3:
        algo_config["tau"] = algo_config_sb3["tau"]
    if "clip_range" in algo_config_sb3:
        algo_config["clip_param"] = algo_config_sb3["clip_range"]
    if "batch_size" in algo_config_sb3:
        algo_config["train_batch_size"] = algo_config_sb3["batch_size"]
    #
    agent = algorithm(
        algo_name = options["algo"], 
        algo_config=algo_config, 
        env_name=options["env_id"],
    )
    #
    iterations = options.get("iterations", 300)
    for i in range(iterations):
        print(f"{options['algo']} iteration nr. {i}             ", end="\r")
        agent.train()

    agent.save_checkpoint(options["save_path"])
    