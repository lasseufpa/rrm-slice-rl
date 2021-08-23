import os
from itertools import product

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from numpy.random import BitGenerator
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
        bs_name: str,
        traffic_types: np.array,
        traffic_throughputs: np.array,
        slice_requirements: dict,
        max_packets_buffer: int = 1024,
        buffer_max_lat: int = 100,
        bandwidth: int = 100000000,
        packet_size: int = 8192 * 8,
        number_ues: int = 10,
        frequency: int = 2,
        total_number_rbs: int = 17,
        max_number_steps: int = 2000,
        max_number_trials: int = 50,
        windows_size: int = 100,
        obs_space_mode: str = "full",
        rng: BitGenerator = np.random.default_rng(),
        plots: bool = False,
    ) -> None:
        self.bs_name = bs_name
        self.max_packets_buffer = max_packets_buffer
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
        self.windows_size = windows_size
        self.obs_space_mode = obs_space_mode
        self.plots = plots
        self.rng = rng

        self.ues, self.slices = self.create_scenario()
        self.action_space_options = self.create_combinations(
            self.total_number_rbs, self.slices.shape[0]
        )
        self.action_space = spaces.Discrete(self.action_space_options.shape[0])

        if self.obs_space_mode == "full":
            self.observation_space = spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    self.ues.shape[0] * 6
                    + self.slices.shape[0] * 9  # Slices
                    + 8  # Slice requirements
                    + self.ues.shape[0],  # UEs spectral efficiency
                ),
            )
        elif self.obs_space_mode == "partial":
            self.observation_space = spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.slices.shape[0] * 9 + 8,),  # Slices + Slice requirements
            )
        else:
            raise Exception(
                'BS observation space mode "{}" is not valid'.format(
                    self.obs_space_mode
                )
            )

        self.hist_labels = [
            "actions",
            "rewards",
        ]
        self.hist = {
            hist_label: np.array([]) if hist_label != "actions" else np.empty((0, 3))
            for hist_label in self.hist_labels
        }

    def step(self, action: int):
        """
        Performs the resource block allocation among slices in according to the
        action received.
        """
        action_values = self.action_space_options[action]
        for i in range(len(action_values)):
            self.slices[i].step(
                self.step_number,
                self.max_number_steps,
                action_values[i],
            )
            if self.step_number == self.max_number_steps - 1:
                self.slices[i].save_hist()

        reward = self.calculate_reward()
        self.update_hist(action_values, reward)
        if self.step_number == self.max_number_steps - 1:
            self.save_hist()
        self.step_number += 1

        return (
            self.get_obs_space(),
            reward,
            self.step_number == (self.max_number_steps),
            {},
        )

    def reset(self, initial_trial: int = -1):
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
            self.trial_number = 1 if initial_trial == -1 else initial_trial
        self.step_number = 0

        self.ues, self.slices = self.create_scenario()
        self.hist_labels = [
            "actions",
            "rewards",
        ]
        self.hist = {
            hist_label: np.array([]) if hist_label != "actions" else np.empty((0, 3))
            for hist_label in self.hist_labels
        }

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
                    bs_name=self.bs_name,
                    id=i,
                    trial_number=self.trial_number,
                    traffic_type=self.traffic_types[i - 1],
                    traffic_throughput=self.traffic_throughputs[i - 1],
                    plots=True,
                    rng=self.rng,
                    windows_size=self.windows_size,
                )
                for i in np.arange(1, self.number_ues + 1)
            ]
        )

        values, indexes = np.unique(self.traffic_types, return_inverse=True)
        # Slices follows an alphabetical order
        slices = np.array(
            [
                Slice(
                    bs_name=self.bs_name,
                    id=i,
                    name=values[i - 1],
                    trial_number=self.trial_number,
                    ues=ues[indexes == (i - 1)],
                    plots=True,
                )
                for i in range(1, len(values) + 1)
            ]
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
        obs_space = (
            np.concatenate(
                (slice_requirements, observation_slices, observation_ues), axis=None
            )
            if self.obs_space_mode == "full"
            else np.concatenate((slice_requirements, observation_slices), axis=None)
        )

        return obs_space

    def calculate_reward(self) -> float:
        """
        Calculates the environment reward for the action taken. It considers
        the slices requirements as basis to formulate how good was the action.
        """
        reward = 0
        for slice in self.slices:
            slice_hist = slice.get_last_hist()
            if slice.name == "embb":
                # Throughput contribution
                reward += (
                    -200
                    if slice_hist["pkt_thr"]
                    < self.mbps_to_packets(
                        self.packet_size,
                        self.slice_requirements["embb"]["throughput"],
                    )
                    else 200
                )
                # Latency contribution
                reward += (
                    100
                    if slice_hist["avg_lat"]
                    <= self.slice_requirements["embb"]["latency"]
                    else -100
                )
                # Packet loss contribution
                reward += (
                    100
                    if slice_hist["pkt_loss"]
                    <= self.slice_requirements["embb"]["pkt_loss"]
                    else -100
                )
            elif slice.name == "urllc":
                # Throughput contribution
                reward += (
                    -100
                    if slice_hist["pkt_thr"]
                    < self.mbps_to_packets(
                        self.packet_size,
                        self.slice_requirements["urllc"]["throughput"],
                    )
                    else 100
                )
                # Latency contribution
                reward += (
                    200
                    if slice_hist["avg_lat"]
                    <= self.slice_requirements["urllc"]["latency"]
                    else -200
                )
                # Packet loss contribution
                reward += (
                    200
                    if slice_hist["pkt_loss"]
                    <= self.slice_requirements["urllc"]["pkt_loss"]
                    else -200
                )
            elif slice.name == "be":
                # Long term average throughput contribution
                reward += (
                    -100
                    if slice_hist["long_term_pkt_thr"]
                    < self.mbps_to_packets(
                        self.packet_size,
                        self.slice_requirements["be"]["long_term_pkt_thr"],
                    )
                    else 100
                )
                # Fifth percentile throughput contribution
                reward += (
                    -100
                    if slice_hist["fifth_perc_pkt_thr"]
                    < self.mbps_to_packets(
                        self.packet_size,
                        self.slice_requirements["be"]["fifth_perc_pkt_thr"],
                    )
                    else 100
                )

        return reward

    def update_ues_traffic(self, traffics: dict) -> None:
        self.traffic_types = traffics

    def create_combinations(self, total_rbs: int, number_slices: int):
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

    def update_hist(self, action_rbs, reward):
        """
        Update the hist values concerned to the basestation.
        """
        self.hist["actions"] = np.vstack((self.hist["actions"], action_rbs))
        self.hist["rewards"] = np.append(self.hist["rewards"], reward, axis=None)

    def save_hist(self) -> None:
        """
        Save variables history to external file.
        """
        path = ("./hist/{}/trial{}/").format(self.bs_name, self.trial_number)
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed(path + "bs", **self.hist)
        if self.plots:
            Basestation.plot_metrics(
                self.bs_name,
                self.trial_number,
                self.slices.shape[0],
                self.ues.shape[0],
            )

    @staticmethod
    def read_hist(bs_name: str, trial_number: int) -> tuple:
        """
        Read variables history from external file.
        """
        path = "./hist/{}/trial{}/bs.npz".format(bs_name, trial_number)
        data = np.load(path)
        return (
            data.f.actions.T,
            data.f.rewards,
        )

    @staticmethod
    def plot_metrics(
        bs_name: str,
        trial_number: int,
        max_slice_id: int,
        step: int = 1,
    ) -> None:
        """
        Plot basestation performance obtained over a specific trial. Read the
        information from external file.
        """

        def plot_slice_metrics():
            filenames = [
                "rcv_thr",
                "snt_thr",
                "pkt_thr_capacity",
                "buffer_occ_rate",
                "avg_buffer_lat",
                "pkt_loss",
                "long_term_pkt_thr",
                "fifth_perc_pkt_thr",
            ]
            x_label = "Iteration [n]"
            y_labels = [
                "Throughput received (Mbps)",
                "Uplink Throughput (Mbps)",
                "Throughput capacity (Mbps)",
                "Occupancy rate",
                "Latency [ms]",
                "Packet loss rate",
                "Long term average thr. (Mbps)",
                "Fifth percentile throughput (Mbps)",
            ]
            slices_name = ["BE", "eMBB", "URLLC"]
            for plot_number in range(len(filenames)):
                w, h = plt.figaspect(0.6)
                fig = plt.figure(figsize=(w, h))
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_labels[plot_number], fontsize=14)
                plt.grid()
                for slice_id in range(1, max_slice_id + 1):
                    hist = Slice.read_hist(bs_name, trial_number, slice_id)[plot_number]
                    hist = (
                        Basestation.packets_to_mbps(8192 * 8, hist)
                        if plot_number in [0, 1, 2, 6, 7]
                        else hist
                    )
                    plt.plot(
                        range(0, len(hist), step),
                        hist[0::step],
                        label="Slice {}".format(slices_name[slice_id - 1]),
                    )
                fig.tight_layout()
                plt.legend(fontsize=12)
                fig.savefig(
                    "./hist/{}/trial{}/{}.pdf".format(
                        bs_name, trial_number, filenames[plot_number]
                    ),
                    # bbox_inches="tight",
                    pad_inches=0,
                    format="pdf",
                    dpi=1000,
                )
                # plt.show()
                plt.close()

        def plot_bs_metrics():
            filenames = [
                "rbs_allocation",
                "rewards",
            ]
            x_label = "Iteration [n]"
            y_labels = [
                "# RBs",
                "Reward",
            ]
            slices_name = ["BE", "eMBB", "URLLC"]
            for plot_number in range(len(filenames)):
                w, h = plt.figaspect(0.6)
                fig = plt.figure(figsize=(w, h))
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_labels[plot_number], fontsize=14)
                hist = Basestation.read_hist(bs_name, trial_number)[plot_number]
                if y_labels[plot_number] == "# RBs":
                    for slice_id in range(0, max_slice_id):
                        plt.plot(
                            range(0, len(hist[slice_id]), step),
                            hist[slice_id][0::step],
                            label="Slice {}".format(slices_name[slice_id]),
                        )
                    plt.legend(fontsize=12)
                else:
                    plt.plot(
                        range(0, len(hist), step),
                        hist[0::step],
                    )
                fig.tight_layout()
                plt.grid()
                fig.savefig(
                    "./hist/{}/trial{}/{}.pdf".format(
                        bs_name, trial_number, filenames[plot_number]
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                    format="pdf",
                    dpi=1000,
                )
                # plt.show()
                plt.close()

        plot_slice_metrics()
        plot_bs_metrics()

    @staticmethod
    def packets_to_mbps(packet_size, number_packets):
        return packet_size * number_packets / 1e6

    @staticmethod
    def mbps_to_packets(packet_size, mbps):
        return np.ceil(mbps * 1e6 / packet_size)


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
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    }
    trials = 2
    basestation = Basestation(
        bs_name="test",
        max_number_trials=trials,
        traffic_types=traffic_types,
        traffic_throughputs=traffic_throughputs,
        slice_requirements=slice_requirements,
        obs_space_mode="partial",
        plots=True,
    )

    basestation.reset()
    for trial in range(1, trials + 1):
        print("Trial ", trial)
        for step_number in tqdm(range(2000)):
            _, _, _, _ = basestation.step(basestation.action_space.sample())
            if step_number == basestation.max_number_steps - 1:
                basestation.reset()


if __name__ == "__main__":
    main()
