import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from baselines import BaselineAgent
from basestation import Basestation

train_param = {
    "steps_per_trial": 2000,
    "total_trials": 45,
    "runs_per_agent": 10,
}

test_param = {
    "steps_per_trial": 2000,
    "total_trials": 50,
    "initial_trial": 50,
    "runs_per_agent": 10,
}

# Create environment
traffics = {
    "light": np.concatenate(
        (
            np.repeat([10], 4),
            np.repeat([1], 3),
            np.repeat([5], 3),
        ),
        axis=None,
    ),
    "moderate": np.concatenate(
        (
            np.repeat([20], 4),
            np.repeat([2], 3),
            np.repeat([10], 3),
        ),
        axis=None,
    ),
    "heavy": np.concatenate(
        (
            np.repeat([30], 4),
            np.repeat([3], 3),
            np.repeat([15], 3),
        ),
        axis=None,
    ),
}
slice_requirements = {
    "light": {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    },
    "moderate": {
        "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 2, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
    },
    "heavy": {
        "embb": {"throughput": 30, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 3, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 15, "fifth_perc_pkt_thr": 5},
    },
}

traffic_types = np.concatenate(
    (
        np.repeat(["embb"], 4),
        np.repeat(["urllc"], 3),
        np.repeat(["be"], 3),
    ),
    axis=None,
)


# Instantiate the agent
def create_agent(type: str, mode: str, obs_space_mode: str, windows_size_obs: int):
    env = Basestation(
        bs_name="dummy",
        max_number_steps=train_param["steps_per_trial"],
        max_number_trials=train_param["total_trials"],
        traffic_types=traffic_types,
        traffic_throughputs=traffics["light"],
        slice_requirements=slice_requirements["light"],
        obs_space_mode=obs_space_mode,
        plots=True,
    )
    if mode == "train":
        if type == "a2c":
            return A2C(
                "MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "ppo":
            return PPO(
                "MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "dqn":
            return DQN(
                "MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard-logs/"
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
        elif type == "mt":
            return BaselineAgent("mt")
        elif type == "pf":
            return BaselineAgent("pf")
        elif type == "rr":
            return BaselineAgent("rr")


models = ["a2c", "ppo", "dqn"]
traffics_list = traffics.keys()
obs_space_modes = ["full", "partial"]
windows_sizes = [1, 50, 100]
seed = 10

# # Training
# print("\n############### Training ###############")
# for windows_size_obs in tqdm(windows_sizes, desc="Windows size", leave=False):
#     for obs_space_mode in tqdm(obs_space_modes, desc="Obs. Space mode", leave=False):
#         for model in tqdm(models, desc="Models", leave=False):
#             rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
#             agent = create_agent(model, "train", obs_space_mode, windows_size_obs)
#             for traffic_behavior in tqdm(traffics_list, desc="Traffics", leave=False):
#                 for run_number in tqdm(
#                     range(1, train_param["runs_per_agent"] + 1), desc="Run", leave=False
#                 ):
#                     env = Basestation(
#                         bs_name="train/{}/ws_{}/{}/{}/run{}".format(
#                             model,
#                             windows_size_obs,
#                             obs_space_mode,
#                             traffic_behavior,
#                             run_number,
#                         ),
#                         max_number_steps=test_param["steps_per_trial"],
#                         max_number_trials=test_param["total_trials"],
#                         traffic_types=traffic_types,
#                         traffic_throughputs=traffics[traffic_behavior],
#                         slice_requirements=slice_requirements[traffic_behavior],
#                         windows_size_obs=windows_size_obs,
#                         obs_space_mode=obs_space_mode,
#                         rng=rng,
#                         plots=True,
#                     )
#                     agent.set_env(env)
#                     agent.learn(
#                         total_timesteps=int(
#                             train_param["total_trials"] * train_param["steps_per_trial"]
#                         ),
#                     )
#             agent.save(
#                 "./agents/{}_{}_ws{}".format(model, obs_space_mode, windows_size_obs)
#             )

# Test
print("\n############### Testing ###############")
# models_test = np.append(models, ["mt", "rr", "pf"])
models_test = ["rr"]
for windows_size_obs in tqdm(windows_sizes, desc="Windows size", leave=False):
    for obs_space_mode in tqdm(obs_space_modes, desc="Obs. Space mode", leave=False):
        for model in tqdm(models_test, desc="Models", leave=False):
            rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
            agent = create_agent(model, "test", obs_space_mode, windows_size_obs)
            for traffic_behavior in tqdm(traffics_list, desc="Traffics", leave=False):
                for run_number in tqdm(
                    range(1, test_param["runs_per_agent"] + 1), desc="Run", leave=False
                ):
                    env = Basestation(
                        bs_name="test/{}/ws_{}/{}/{}/run{}".format(
                            model,
                            windows_size_obs,
                            obs_space_mode,
                            traffic_behavior,
                            run_number,
                        ),
                        max_number_steps=test_param["steps_per_trial"],
                        max_number_trials=test_param["total_trials"],
                        traffic_types=traffic_types,
                        traffic_throughputs=traffics[traffic_behavior],
                        slice_requirements=slice_requirements[traffic_behavior],
                        windows_size_obs=windows_size_obs,
                        obs_space_mode=obs_space_mode,
                        rng=rng,
                        plots=True,
                    )
                    agent.set_env(env)
                    obs = env.reset(test_param["initial_trial"])
                    for _ in tqdm(
                        range(
                            test_param["total_trials"] - test_param["initial_trial"] + 1
                        ),
                        leave=False,
                        desc="Trials",
                    ):
                        for _ in tqdm(
                            range(test_param["steps_per_trial"]),
                            leave=False,
                            desc="Steps",
                        ):
                            action, _states = agent.predict(obs)
                            obs, rewards, dones, info = env.step(action)
                        env.reset()
