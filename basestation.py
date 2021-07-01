from itertools import product

import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from slice import Slice
from ue import UE


class Basestation(gym.Env):
    """
    Basestation class containing the Gym environment variables and functions to
    perform the radio resource management of the basestation in accordance with
    the UEs and slices defined. It receives the action provided by the RL agent
    and applies it to RRM function. The observation space is composed by slices
    and UEs observations. The main() function implements a random agent
    demostrating how the environment can be used outside a RL library.
    """

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
        max_number_trials: int,
        traffic_types: np.array,
        traffic_throughputs: np.array,
        slice_requirements: dict,
    ):
        self.buffer_size = buffer_size
        self.buffer_max_lat = buffer_max_lat
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.number_ues = number_ues
        self.frequency = frequency
        self.total_number_rbs = total_number_rbs
        self.max_number_steps = max_number_steps
        self.max_number_trials = max_number_trials
        self.traffic_types = traffic_types
        self.step_number = 0
        self.trial_number = 1
        self.reward = 0
        self.traffic_throughputs = traffic_throughputs
        self.slice_requirements = slice_requirements

        self.ues, self.slices = self.create_scenario()
        self.action_space_options = self.create_combinations(
            self.total_number_rbs, self.slices.shape[0]
        )
        self.action_space = spaces.Discrete(self.action_space_options.shape[0])
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(
                (self.ues.shape[0] + self.slices.shape[0]) * 6
                + len(self.slice_requirements) * 3,
            ),
        )

    def step(self, action):
        """
        Performs the resource block allocation among slices in according to the
        action received.
        """
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
            self.step_number == (self.max_number_steps),
            {},
        )

    def reset(self):
        """
        Reset the UEs and Slices to enable the environment to start other
        episode without past residuous. The reset function increases
        the number of trials when a trial is finished.
        """
        if (
            self.step_number == self.max_number_steps
            and self.trial_number < self.max_number_trials
        ):
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
        """
        Creates UEs and slices as specified in the basestation init.
        """
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
                    self.traffic_throughputs[i - 1],
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
        """
        Get observation space variable that is composed by slices and UEs
        information.
        """
        slice_requirements = np.array([])
        observation_slices = np.array([])
        observation_ues = np.array([])

        for slice_req in self.slice_requirements:
            for attribute in self.slice_requirements[slice_req]:
                slice_requirements = np.append(
                    slice_requirements, self.slice_requirements[slice_req][attribute]
                )

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

        return np.concatenate(
            (slice_requirements, observation_slices, observation_ues), axis=None
        )

    def calculate_reward(self):
        """
        Calculates the environment reward for the action taken. It considers
        the slices requirements as basis to formulate how good was the action.
        """
        reward = 0
        slice_labels = ["embb", "urllc", "be"]
        for i, slice in enumerate(self.slices):
            slice_hist = slice.get_last_hist()
            # Throughput contribution
            reward += (
                -100
                if min(slice_hist["pkt_snt"], slice_hist["pkt_thr"])
                < self.mpbs_to_packets(
                    self.slice_requirements[slice_labels[i]]["throughput"]
                )
                else 100
            )
            # Latency contribution
            reward += (
                -100
                if slice_hist["avg_lat"]
                < self.slice_requirements[slice_labels[i]]["latency"]
                else 100
            )
            # Dropped packets contribution
            reward += (
                -100
                if slice_hist["dropped_pkts"]
                < self.slice_requirements[slice_labels[i]]["dropped_packets"]
                else 100
            )

        return reward

    def create_combinations(self, total_rbs, number_slices):
        """
        Create the combinations of possible arrays with RBs allocation for each
        slice. For instance, let's assume 3 slices and 17 RBs available in the
        basestation, a valid array should be [1, 13, 3] since its summation is
        equal to 17 RBs. Moreover, it indicates that the first slice received 1
        RB, the second received 13 RBs, and the third received 3 RBs. A valid
        array always has a summation equal to the total number of RBs in a
        basestation and has its array-length equal to the number of slices. An
        action taken by RL agent is a discrete number that represents the index
        of the option into the array with all possible RBs allocations for
        these slices.
        """
        combinations = []
        combs = product(range(0, total_rbs + 1), repeat=number_slices)
        for comb in combs:
            if np.sum(comb) == total_rbs:
                combinations.append(comb)
        return np.asarray(combinations)

    def packets_to_mbps(self, number_packets) -> float:
        return self.packet_size * number_packets / 1e6

    def mpbs_to_packets(self, mbps) -> int:
        return np.ceil(mbps / (self.packet_size))


def main():
    # Random agent implementation
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
    basestation = Basestation(
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
