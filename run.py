import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from basestation import Basestation

# Create environment
traffic_types = np.concatenate(
    (
        np.repeat(["embb"], 4),
        np.repeat(["urllc"], 3),
        np.repeat(["be"], 3),
    ),
    axis=None,
)
traffic_throughputs = np.concatenate(
    (
        np.repeat([10], 4),
        np.repeat([0.6], 3),
        np.repeat([5], 3),
    ),
    axis=None,
)
slice_requirements = {
    "embb": {"throughput": 10, "latency": 10, "dropped_packets": 100},
    "urllc": {"throughput": 0.6, "latency": 1, "dropped_packets": 0},
    "be": {"throughput": 5, "latency": 100, "dropped_packets": 100},
}
env = Basestation(
    100 * 8192 * 8,
    100,
    5000000,
    8192 * 8,
    10,
    1,
    17,
    2000,
    2,
    traffic_types,
    traffic_throughputs,
    slice_requirements,
    True,
)
# check_env(env)
# exit()

# Instantiate the agent
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    verbose=1,
    tensorboard_log="./tensorboard-logs/",
)
# Train the agent
model.learn(total_timesteps=int(4000))
# Save the agent
model.save("dqn_rrm")
del model  # delete trained model to demonstrate loading

# Load the trained agent
env = Basestation(
    100 * 8192 * 8,
    100,
    5000000,
    8192 * 8,
    10,
    1,
    17,
    2000,
    2,
    traffic_types,
    traffic_throughputs,
    slice_requirements,
    True,
)
model = A2C.load("dqn_rrm", env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)

# # Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
