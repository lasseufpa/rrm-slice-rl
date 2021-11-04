import os

import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from baselines import BaselineAgent
from basestation import Basestation
from callbacks import ProgressBarManager

train_param = {
    "steps_per_trial": 2000,
    "total_trials": 49,
    "runs_per_agent": 10,
}

test_param = {
    "steps_per_trial": 2000,
    "total_trials": 50,
    "initial_trial": 50,
    "runs_per_agent": 1,
}

# Create environment
traffic_types = np.concatenate(
    (
        np.repeat(["embb"], 4),
        np.repeat(["urllc"], 3),
        np.repeat(["be"], 3),
    ),
    axis=None,
)
traffic_throughputs = {
    "light": {
        "embb": 15,
        "urllc": 1,
        "be": 5,
    },
    # "moderate": {
    #     "embb": 25,
    #     "urllc": 5,
    #     "be": 10,
    # },
}
slice_requirements_traffics = {
    "light": {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    },
    # "moderate": {
    #     "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
    #     "urllc": {"throughput": 5, "latency": 1, "pkt_loss": 0.001},
    #     "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
    # },
}

models = ["sac", "td3"]  # ["ppo", "sac", "td3", "ppo"]
traffics_list = traffic_throughputs.keys()
# obs_space_modes = ["full", "partial"]
obs_space_modes = ["full"]
# windows_sizes = [1, 10, 50]
windows_sizes = [10]
seed = 10
model_save_freq = int(
    train_param["total_trials"]
    * train_param["steps_per_trial"]
    * train_param["runs_per_agent"]
    / 10
)
n_eval_episodes = 5  # default is 5
eval_freq = 10000  # default is 10000


# Instantiate the agent
def create_agent(
    type: str, env: VecNormalize, mode: str, obs_space_mode: str, windows_size_obs: int
):
    if mode == "train":
        if type == "a2c":
            return A2C(
                "MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "ppo":
            return PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
            )
        elif type == "dqn":
            return DQN(
                "MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "sac":
            return SAC(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                policy_kwargs=dict(net_arch=[400, 300]),
            )
        elif type == "td3":
            return TD3(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                policy_kwargs=dict(net_arch=[400, 300]),
            )
    elif mode == "test":
        if type == "a2c":
            return A2C.load(
                "./agents/a2c_{}_ws{}".format(obs_space_mode, windows_size_obs),
                None,
                verbose=0,
            )
        elif type == "ppo":
            return PPO.load(
                "./agents/ppo_{}_ws{}".format(obs_space_mode, windows_size_obs),
                None,
                verbose=0,
            )
        elif type == "dqn":
            return DQN.load(
                "./agents/dqn_{}_ws{}".format(obs_space_mode, windows_size_obs),
                None,
                verbose=0,
            )
        elif type == "sac":
            return SAC.load(
                "./agents/sac_{}_ws{}".format(obs_space_mode, windows_size_obs),
                None,
                verbose=0,
            )
        elif type == "td3":
            return TD3.load(
                "./agents/td3_{}_ws{}".format(obs_space_mode, windows_size_obs),
                None,
                verbose=0,
            )
        elif type == "mt":
            return BaselineAgent("mt")
        elif type == "pf":
            return BaselineAgent("pf")
        elif type == "rr":
            return BaselineAgent("rr")


# Removing VecNormalize models from previous simulations
dir_vec_models = "./vecnormalize_models"
if not os.path.exists(dir_vec_models):
    os.makedirs(dir_vec_models)
for f in os.listdir(dir_vec_models):
    os.remove(os.path.join(dir_vec_models, f))

# Training
print("\n############### Training ###############")
for windows_size_obs in tqdm(windows_sizes, desc="Windows size", leave=False):
    for obs_space_mode in tqdm(obs_space_modes, desc="Obs. Space mode", leave=False):
        for model in tqdm(models, desc="Models", leave=False):
            rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
            env = Basestation(
                bs_name="train/{}/ws_{}/{}/".format(
                    model,
                    windows_size_obs,
                    obs_space_mode,
                ),
                max_number_steps=train_param["steps_per_trial"],
                max_number_trials=train_param["total_trials"],
                traffic_types=traffic_types,
                traffic_throughputs=traffic_throughputs,
                slice_requirements_traffics=slice_requirements_traffics,
                windows_size_obs=windows_size_obs,
                obs_space_mode=obs_space_mode,
                rng=rng,
            )
            env = DummyVecEnv([lambda: env])
            dir_vec_file = dir_vec_models + "/{}_{}_ws{}.pkl".format(
                model, obs_space_mode, windows_size_obs
            )
            env = VecNormalize(env)
            agent = create_agent(model, env, "train", obs_space_mode, windows_size_obs)
            agent.set_random_seed(seed)
            callback_checkpoint = CheckpointCallback(
                save_freq=model_save_freq,
                save_path="./agents/",
                name_prefix="{}_{}_ws{}".format(
                    model, obs_space_mode, windows_size_obs
                ),
            )
            callback_evaluation = EvalCallback(
                eval_env=env,
                log_path="./evaluations/",
                best_model_save_path="./agents/best_{}_{}_ws{}/".format(
                    model, obs_space_mode, windows_size_obs
                ),
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                verbose=False,
                warn=False,
            )
            with ProgressBarManager(
                int(
                    train_param["total_trials"]
                    * train_param["steps_per_trial"]
                    * train_param["runs_per_agent"]
                )
            ) as callback_progress_bar:
                agent.learn(
                    total_timesteps=int(
                        train_param["total_trials"]
                        * train_param["steps_per_trial"]
                        * train_param["runs_per_agent"]
                    ),
                    callback=[
                        callback_progress_bar,
                        callback_checkpoint,
                        callback_evaluation,
                    ],
                )
            env.save(dir_vec_file)
            agent.save(
                "./agents/{}_{}_ws{}".format(model, obs_space_mode, windows_size_obs)
            )

# Test
print("\n############### Testing ###############")
models_test = np.append(models, ["mt", "rr", "pf"])
for windows_size_obs in tqdm(windows_sizes, desc="Windows size", leave=False):
    for obs_space_mode in tqdm(obs_space_modes, desc="Obs. Space mode", leave=False):
        for model in tqdm(models_test, desc="Models", leave=False):
            rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
            env = Basestation(
                bs_name="test/{}/ws_{}/{}/".format(
                    model,
                    windows_size_obs,
                    obs_space_mode,
                ),
                max_number_steps=test_param["steps_per_trial"],
                max_number_trials=test_param["total_trials"],
                traffic_types=traffic_types,
                traffic_throughputs=traffic_throughputs,
                slice_requirements_traffics=slice_requirements_traffics,
                windows_size_obs=windows_size_obs,
                obs_space_mode=obs_space_mode,
                rng=rng,
                plots=True,
                save_hist=True,
                baseline=False if model in models else True,
            )
            obs = (
                [env.reset(test_param["initial_trial"])]
                if model in models
                else env.reset(test_param["initial_trial"])
            )

            if model in models:
                dir_vec_models = "./vecnormalize_models"
                dir_vec_file = dir_vec_models + "/{}_{}_ws{}.pkl".format(
                    model, obs_space_mode, windows_size_obs
                )
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(dir_vec_file, env)
                env.training = False
                env.norm_reward = False
            agent = create_agent(model, env, "test", obs_space_mode, windows_size_obs)
            agent.set_random_seed(seed)
            for _ in tqdm(
                range(test_param["total_trials"] + 1 - test_param["initial_trial"]),
                leave=False,
                desc="Trials",
            ):
                for _ in tqdm(
                    range(test_param["steps_per_trial"]),
                    leave=False,
                    desc="Steps",
                ):
                    action, _states = (
                        agent.predict(obs, deterministic=True)
                        if model in models
                        else agent.predict(obs)
                    )
                    obs, rewards, dones, info = env.step(action)
                env.reset()
