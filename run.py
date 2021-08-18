import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from baselines import BaselineAgent
from basestation import Basestation

train_param = {
    "steps_per_trial": 2000,
    "total_trials": 2,
    "runs_per_agent": 2,
}

test_param = {
    "steps_per_trial": 2000,
    "total_trials": 50,
    "initial_trial": 50,
    "runs_per_agent": 2,
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
    "heavy": np.concatenate(
        (
            np.repeat([20], 4),
            np.repeat([2], 3),
            np.repeat([10], 3),
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
    "heavy": {
        "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 2, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
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
def create_agent(type: str, mode: str):
    env = Basestation(
        bs_name="dummy",
        max_number_steps=train_param["steps_per_trial"],
        max_number_trials=train_param["total_trials"],
        traffic_types=traffic_types,
        traffic_throughputs=traffics["light"],
        slice_requirements=slice_requirements["light"],
        plots=True,
    )
    if mode == "train":
        if type == "a2c":
            return A2C(
                "MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "ppo":
            return PPO(
                "MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/"
            )
        elif type == "dqn":
            return DQN(
                "MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/"
            )
    elif mode == "test":
        if type == "a2c":
            return A2C.load("./agents/a2c", env)
        elif type == "ppo":
            return PPO.load("./agents/ppo", env)
        elif type == "dqn":
            return DQN.load("./agents/dqn", env)
        elif type == "mt":
            return BaselineAgent("mt")
        elif type == "pf":
            return BaselineAgent("pf")
        elif type == "rr":
            return BaselineAgent("rr")


models = ["a2c"]  # , "ppo", "dqn"]
traffics_list = traffics.keys()
seed = 10
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()

# Training
# for model in models:
#     agent = create_agent(model, "train")
#     for traffic_behavior in traffics_list:
#         for run_number in range(1, train_param["runs_per_agent"] + 1):
#             env = Basestation(
#                 bs_name="{}_train/{}/run{}".format(model, traffic_behavior, run_number),
#                 max_number_steps=train_param["steps_per_trial"],
#                 max_number_trials=train_param["total_trials"],
#                 traffic_types=traffic_types,
#                 traffic_throughputs=traffics[traffic_behavior],
#                 slice_requirements=slice_requirements[traffic_behavior],
#                 windows_size = 100,
#                 seed=rng.random(),
#                 plots=True,
#             )
#             agent.set_env(env)
#             agent.learn(
#                 total_timesteps=int(
#                     train_param["total_trials"] * train_param["steps_per_trial"]
#                 ),
#             )
#     agent.save("./agents/{}".format(model))

# Test
# models_test = np.append(models, ["mt", "rr", "pf"])
models_test = ["pf"]  # , "rr", "mt"]
for model in models_test:
    agent = create_agent(model, "test")
    for traffic_behavior in traffics_list:
        for run_number in range(1, test_param["runs_per_agent"] + 1):
            env = Basestation(
                bs_name="{}_test/{}/run{}".format(model, traffic_behavior, run_number),
                max_number_steps=test_param["steps_per_trial"],
                max_number_trials=test_param["total_trials"],
                traffic_types=traffic_types,
                traffic_throughputs=traffics[traffic_behavior],
                slice_requirements=slice_requirements[traffic_behavior],
                windows_size=100,
                seed=rng.random(),
                plots=True,
            )
            agent.set_env(env)
            obs = env.reset(test_param["initial_trial"])
            for _ in range(
                test_param["total_trials"] - test_param["initial_trial"] + 1
            ):
                for _ in range(test_param["steps_per_trial"]):
                    action, _states = agent.predict(obs)
                    obs, rewards, dones, info = env.step(action)
                env.reset()
