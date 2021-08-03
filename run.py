import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from basestation import Basestation

train_param = {
    "total_trials": 45,
    "train_init_trial": 1,
    "train_repeat": 10,
    "runs_per_agent": 1,
}

test_param = {
    "total_trials": 50,
    "train_init_trial": 45,
    "train_repeat": 10,
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
traffic_types = np.concatenate(
    (
        np.repeat(["embb"], 4),
        np.repeat(["urllc"], 3),
        np.repeat(["be"], 3),
    ),
    axis=None,
)
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


# Instantiate the agent
def create_agent(type: str, env: Basestation):
    if type == "a2c":
        return A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/")
    elif type == "ppo":
        return PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/")
    elif type == "dqn":
        return DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard-logs/")


models = ["a2c"]  # , "ppo", "dqn"]

# Training
# for model in models:
#     agent = create_agent(model)
#     for run_number in range(train_param["runs_per_agent"]):
#         env = Basestation(
#             "{}_run_{}".format(model[0], run_number),
#             1024 * 8192 * 8,
#             100,
#             100000000,
#             8192 * 8,
#             10,
#             2,
#             17,
#             2000,
#             trials,
#             traffic_types,
#             traffics["light"],
#             slice_requirements["light"],
#             True,
#         )
#         model[1].model.learn(total_timesteps=int(trials * 2000 * run_number))
#     model[0].save("./agents/{}_agent".format(model[0]))


# del model  # delete trained model to demonstrate loading

# # Load the trained agent
# env = Basestation(
#     "test",
#     1024 * 8192 * 8,
#     100,
#     100000000,
#     8192 * 8,
#     10,
#     2,
#     17,
#     2000,
#     trials,
#     traffic_types,
#     traffic_throughputs,
#     slice_requirements,
#     True,
# )
# model = A2C.load("dqn_rrm", env)

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)

# # Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
