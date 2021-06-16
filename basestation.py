from itertools import product

import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from slice import Slice
from ue import UE


class Basestation(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        buffer_size: int,
        buffer_max_lat: int,
        bandwidth: int,
        packet_size: int,
        number_ues: int,
        frequency: int,
        total_number_rbs: int,
        max_number_steps: int,
        traffic_types: np.array,
    ):
        self.buffer_size = buffer_size
        self.buffer_max_lat = buffer_max_lat
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.number_ues = number_ues
        self.frequency = frequency
        self.total_number_rbs = total_number_rbs
        self.max_number_steps = max_number_steps
        self.traffic_types = traffic_types
        self.step_number = 0
        self.trial_number = 1
        self.reward = 0

        self.ues, self.slices = self.create_scenario()
        self.action_space_options = self.create_combinations(
            self.total_number_rbs, self.slices.shape[0]
        )
        self.action_space = spaces.Discrete(self.action_space_options.shape[0])
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=((self.ues.shape[0] + self.slices.shape[0]) * 6,)
        )

    def step(self, action):
        action_values = self.action_space_options[action]
        for i in range(len(action_values)):
            self.slices[i].step(
                self.step_number, self.max_number_steps, action_values[i]
            )
            if self.step_number == self.max_number_steps - 1:
                self.slices[i].save_hist(self.trial_number)
        self.step_number += 1

        return (
            self.get_obs_space(),
            self.calculate_reward(),
            self.step_number == (self.max_number_steps - 1),
            {},
        )

    def reset(self):
        if self.step_number == self.max_number_steps:
            self.trial_number += 1
        else:
            self.trial_number = 1
        self.step_number = 0

        self.ues, self.slices = self.create_scenario()

        return self.get_obs_space()

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def create_scenario(self):
        ues = np.array(
            [
                UE(
                    i,
                    self.buffer_size,
                    self.buffer_max_lat,
                    self.bandwidth,
                    self.packet_size,
                    self.trial_number,
                    self.traffic_types[i - 1],
                    self.frequency,
                    self.total_number_rbs,
                )
                for i in np.arange(1, self.number_ues + 1)
            ]
        )

        values, indexes = np.unique(self.traffic_types, return_inverse=True)
        # Slices follows an alphabetical order
        slices = np.array(
            [Slice(i, ues[indexes == (i - 1)]) for i in range(1, len(values) + 1)]
        )

        return ues, slices

    def get_obs_space(self):
        observation_slices = np.array([])
        observation_ues = np.array([])
        for slice in self.slices:
            for array in slice.hist.values():
                observation_slices = np.append(
                    observation_slices, (array[-1] if len(array) != 0 else 0)
                )

            for ue in slice.ues:
                for array in ue.hist.values():
                    observation_ues = np.append(
                        observation_ues, (array[-1] if len(array) != 0 else 0)
                    )

        return np.append(observation_slices, observation_ues)

    def calculate_reward(self):
        return 10  # TODO

    def create_combinations(self, total_rbs, number_slices):
        combinations = []
        combs = product(range(0, total_rbs + 1), repeat=number_slices)
        for comb in combs:
            if np.sum(comb) == total_rbs:
                combinations.append(comb)
        return np.asarray(combinations)


def main():
    traffic_types = np.concatenate(
        (np.repeat("embb", 4), np.repeat("urllc", 3), np.repeat("be", 3)), axis=None
    )
    basestation = Basestation(
        10 * 65535 * 8, 100, 5000000, 65535 * 8, 10, 1, 17, 2000, traffic_types
    )
    trials = 2

    basestation.reset()
    for trial in range(1, trials + 1):
        print("Trial ", trial)
        for step_number in tqdm(range(2000)):
            _, _, _, _ = basestation.step(basestation.action_space.sample())
            if step_number == basestation.max_number_steps - 1:
                basestation.reset()


if __name__ == "__main__":
    main()
