import os

import joblib
import numpy as np
from stable_baselines3 import SAC, TD3, PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from baselines import BaselineAgent
from basestation import Basestation
from callbacks import ProgressBarManager

train_param = {
    "steps_per_trial": 2000,
    "total_trials": 45,
    "runs_per_agent": 10,
}

test_param = {
    "steps_per_trial": 2000,
    "total_trials": 50,
    "initial_trial": 46,
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
        "be": 15,
    },
    "moderate": {
        "embb": 25,
        "urllc": 5,
        "be": 25,
    },
}
slice_requirements_traffics = {
    "light": {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 1e-5},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    },
    "moderate": {
        "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 5, "latency": 1, "pkt_loss": 1e-5},
        "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
    },
}

models = ["intentless", "colran", "sac"]
obs_space_modes = ["full", "partial"]
windows_sizes = [1]  # , 50, 100]
seed = 100
model_save_freq = int(
    train_param["total_trials"]
    * train_param["steps_per_trial"]
    * train_param["runs_per_agent"]
    / 10
)
n_eval_episodes = 5  # default is 5
eval_freq = 10000  # default is 10000
test_model = "best"  # or last


# Instantiate the agent
def create_agent(
    type: str,
    env: VecNormalize,
    mode: str,
    obs_space_mode: str,
    windows_size_obs: int,
    test_model: str = "best",
):
    def optimized_hyperparameters(model: str, obs_space: str):
        hyperparameters = joblib.load(
            "hyperparameter_opt/{}_{}_ws{}.pkl".format(
                model, obs_space, windows_size_obs
            )
        ).best_params
        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }[hyperparameters["net_arch"]]
        hyperparameters["policy_kwargs"] = dict(net_arch=net_arch)
        hyperparameters.pop("net_arch")
        hyperparameters["target_entropy"] = "auto"
        hyperparameters["ent_coef"] = "auto"
        hyperparameters["gradient_steps"] = hyperparameters["train_freq"]

        return hyperparameters

    if mode == "train":
        if type == "sac":
            hyperparameters = optimized_hyperparameters(type, obs_space_mode)
            return SAC(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                **hyperparameters,
                seed=seed,
            )
        elif type == "td3":
            hyperparameters = optimized_hyperparameters(type, obs_space_mode)
            return TD3(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                **hyperparameters,
                seed=seed,
            )
        elif type == "intentless":
            return DDPG(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                seed=seed,
            )
        elif type == "colran":
            return PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                seed=seed,
            )
    elif mode == "test":
        path = (
            "./agents/best_{}_{}_ws{}/best_model".format(
                type, obs_space_mode, windows_size_obs
            )
            if test_model == "best"
            else "./agents/{}_{}_ws{}".format(type, obs_space_mode, windows_size_obs)
        )
        if type == "sac":
            return SAC.load(
                path,
                None,
                verbose=0,
            )
        elif type == "td3":
            return TD3.load(
                path,
                None,
                verbose=0,
            )
        elif type == "intentless":
            return DDPG.load(
                path,
                None,
                verbose=0,
            )
        elif type == "colran":
            return PPO.load(
                path,
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
                agent_type="main" if model not in ["intentless", "colran"] else model,
            )
            env = Monitor(env)
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

            if model in models:
                dir_vec_models = "./vecnormalize_models"
                dir_vec_file = dir_vec_models + "/{}_{}_ws{}.pkl".format(
                    model, obs_space_mode, windows_size_obs
                )
                env = Monitor(env)
                dict_reset = {"initial_trial": test_param["initial_trial"]}
                obs = [env.reset(**dict_reset)]
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(dir_vec_file, env)
                env.training = False
                env.norm_reward = False
            elif not (model in models):
                obs = env.reset(test_param["initial_trial"])
            agent = create_agent(
                model, env, "test", obs_space_mode, windows_size_obs, test_model
            )
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
                if model not in models:
                    env.reset()
