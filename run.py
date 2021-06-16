import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from basestation import Basestation

# Create environment
traffic_types = np.concatenate(
    (np.repeat("embb", 4), np.repeat("urllc", 3), np.repeat("be", 3)), axis=None
)
env = Basestation(
    10 * 65535 * 8, 100, 5000000, 65535 * 8, 10, 1, 17, 2000, traffic_types
)
# check_env(env)
# exit()

# Instantiate the agent
model = DQN(
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
    10 * 65535 * 8, 100, 5000000, 65535 * 8, 10, 1, 17, 2000, traffic_types
)
model = DQN.load("dqn_rrm", env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
