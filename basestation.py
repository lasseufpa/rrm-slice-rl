import os
from itertools import product

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from numpy.random import BitGenerator
from numpy.testing._private.utils import requires_memory
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
        slice_requirements_traffics: dict,
        max_packets_buffer: int = 1024,
        buffer_max_lat: int = 100,
        bandwidth: int = 100000000,
        packet_size: int = 8192 * 8,
        number_ues: int = 10,
        frequency: int = 2,
        total_number_rbs: int = 17,
        max_number_steps: int = 2000,
        max_number_trials: int = 50,
        windows_size_obs: int = 100,
        steps_update_traffics: int = 200,
        obs_space_mode: str = "full",
        rng: BitGenerator = np.random.default_rng(),
        plots: bool = False,
        slice_plots: bool = False,
        ue_plots: bool = False,
        save_hist: bool = False,
        normalize_ue_obs: bool = False,
        baseline: bool = False,
        root_path: str = ".",
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
        self.slice_requirements_traffics = slice_requirements_traffics
        self.slice_requirements = self.slice_requirements_traffics[
            list(self.slice_requirements_traffics.keys())[0]
        ]
        self.windows_size_obs = windows_size_obs
        self.steps_update_traffics = steps_update_traffics
        self.obs_space_mode = obs_space_mode
        self.plots = plots
        self.slice_plots = slice_plots
        self.ue_plots = ue_plots
        self.save_hist_bool = save_hist
        self.normalize_ue_obs = normalize_ue_obs
        self.root_path = root_path
        self.rng = rng

        self.ues, self.slices = self.create_scenario()
        self.action_space_options = self.create_combinations(
            self.total_number_rbs, self.slices.shape[0], baseline
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.slices.shape[0],))

        if self.obs_space_mode == "full":
            self.observation_space = spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    self.ues.shape[0] * np.sum(len(self.ues[0].hist.keys()))  # UEs
                    + self.slices.shape[0]
                    * np.sum(len(self.slices[0].hist.keys()))  # Slices
                    + np.sum(
                        np.fromiter(
                            (len(item) for item in self.slice_requirements.values()),
                            int,
                        )
                    ),  # Slice requirements
                ),
                dtype=np.float32,
            )
        elif self.obs_space_mode == "partial":
            self.observation_space = spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    self.slices.shape[0] * np.sum(len(self.slices[0].hist.keys()))
                    + np.sum(
                        np.fromiter(
                            (len(item) for item in self.slice_requirements.values()),
                            int,
                        )
                    ),
                ),  # Slices + Slice requirements
                dtype=np.float32,
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
        self.slice_req_norm_factors = (
            [
                100,
                self.buffer_max_lat,
                1,
                100,
                self.buffer_max_lat,
                1,
                100,
                100,
            ]
            if self.normalize_ue_obs
            else [1, 1, 1, 1, 1, 1, 1, 1]
        )

    def step(self, action: np.array):
        """
        Performs the resource block allocation among slices in according to the
        action received.
        """
        rbs_allocation = (
            ((action + 1) / np.sum(action + 1)) * self.total_number_rbs
            if np.sum(action + 1) != 0
            else np.ones(action.shape[0])
            * (1 / action.shape[0])
            * self.total_number_rbs
        )
        action_idx = np.argmin(
            np.sum(np.abs(self.action_space_options - rbs_allocation), axis=1)
        )
        action_values = self.action_space_options[action_idx]
        for i in range(len(action_values)):
            self.slices[i].step(
                self.step_number,
                self.max_number_steps,
                action_values[i],
            )
            if (self.step_number == self.max_number_steps - 1) and self.save_hist_bool:
                self.slices[i].save_hist()

        reward = self.calculate_reward()
        self.update_hist(action_values, reward)
        if (self.step_number == self.max_number_steps - 1) and self.save_hist_bool:
            self.save_hist()
        self.step_number += 1
        if self.step_number % self.steps_update_traffics == 0:
            self.update_ues_traffic()

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
        if (self.step_number == 0 and self.trial_number == 1) or (
            self.trial_number == self.max_number_trials
        ):
            self.trial_number = 1 if initial_trial == -1 else initial_trial
        elif self.trial_number < self.max_number_trials:
            self.trial_number += 1
        else:
            raise Exception(
                "Trial number received a non expected value equals to {}.".format(
                    self.trial_number
                )
            )
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
                    traffic_throughput=self.traffic_throughputs[
                        list(self.traffic_throughputs.keys())[0]
                    ][self.traffic_types[i - 1]],
                    plots=self.ue_plots,
                    rng=self.rng,
                    windows_size_obs=self.windows_size_obs,
                    normalize_obs=self.normalize_ue_obs,
                    root_path=self.root_path,
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
                    plots=self.slice_plots,
                    save_hist=self.save_hist_bool,
                    root_path=self.root_path,
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

        normalization_idx = 0
        for slice_req in self.slice_requirements:
            for attribute in self.slice_requirements[slice_req]:
                slice_requirements = np.append(
                    slice_requirements,
                    (
                        self.slice_requirements[slice_req][attribute]
                        / self.slice_req_norm_factors[normalization_idx]
                    ),
                )
                normalization_idx += 1

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
        w_embb_thr = 0.2
        w_embb_lat = 0.05
        w_embb_loss = 0.05
        w_urllc_thr = 0.1
        w_urllc_lat = 0.25
        w_urllc_loss = 0.25
        w_be_long = 0.05
        w_be_fifth = 0.05
        reward = 0
        for slice in self.slices:
            slice_hist = slice.get_last_no_windows_hist()
            if slice.name == "embb":
                req_thr_normalized = (
                    self.slice_requirements["embb"]["throughput"]
                    / self.slice_req_norm_factors[0]
                )
                req_lat_normalized = (
                    self.slice_requirements["embb"]["latency"]
                    / self.slice_req_norm_factors[1]
                )
                req_pkt_loss_normalized = (
                    self.slice_requirements["embb"]["pkt_loss"]
                    / self.slice_req_norm_factors[2]
                )

                # Throughput contribution
                reward += (
                    -w_embb_thr
                    * (
                        (req_thr_normalized - slice_hist["pkt_thr"])
                        / req_thr_normalized
                    )
                    if slice_hist["pkt_thr"] < req_thr_normalized
                    else 0
                )
                # Latency contribution
                reward += (
                    0
                    if slice_hist["avg_lat"] <= req_lat_normalized
                    else -w_embb_lat
                    * (slice_hist["avg_lat"] - req_lat_normalized)
                    / (
                        (self.buffer_max_lat / self.slice_req_norm_factors[1])
                        - req_lat_normalized
                    )
                )
                # Packet loss contribution
                reward += (
                    0
                    if slice_hist["pkt_loss"] <= req_pkt_loss_normalized
                    else -w_embb_loss
                    * (slice_hist["pkt_loss"] - req_pkt_loss_normalized)
                    / (1 - req_pkt_loss_normalized)
                )
            elif slice.name == "urllc":
                req_thr_normalized = (
                    self.slice_requirements["urllc"]["throughput"]
                    / self.slice_req_norm_factors[3]
                )
                req_lat_normalized = (
                    self.slice_requirements["urllc"]["latency"]
                    / self.slice_req_norm_factors[4]
                )
                req_pkt_loss_normalized = (
                    self.slice_requirements["urllc"]["pkt_loss"]
                    / self.slice_req_norm_factors[5]
                )

                # Throughput contribution
                reward += (
                    -w_urllc_thr
                    * (
                        (req_thr_normalized - slice_hist["pkt_thr"])
                        / req_thr_normalized
                    )
                    if slice_hist["pkt_thr"] < req_thr_normalized
                    else 0
                )
                # Latency contribution
                reward += (
                    0
                    if slice_hist["avg_lat"] <= req_lat_normalized
                    else -w_urllc_lat
                    * (slice_hist["avg_lat"] - req_lat_normalized)
                    / (
                        (self.buffer_max_lat / self.slice_req_norm_factors[1])
                        - req_lat_normalized
                    )
                )
                # Packet loss contribution
                reward += (
                    0
                    if slice_hist["pkt_loss"] <= req_pkt_loss_normalized
                    else -w_urllc_loss
                    * (slice_hist["pkt_loss"] - req_pkt_loss_normalized)
                    / (1 - req_pkt_loss_normalized)
                )
            elif slice.name == "be":
                req_long_thr_normalized = (
                    self.slice_requirements["be"]["long_term_pkt_thr"]
                    / self.slice_req_norm_factors[6]
                )
                req_fifth_thr_normalized = (
                    self.slice_requirements["be"]["fifth_perc_pkt_thr"]
                    / self.slice_req_norm_factors[7]
                )

                # Long term average throughput contribution
                reward += (
                    -w_be_long
                    * (
                        (req_long_thr_normalized - slice_hist["long_term_pkt_thr"])
                        / req_long_thr_normalized
                    )
                    if slice_hist["long_term_pkt_thr"] < req_long_thr_normalized
                    else 0
                )
                # Fifth percentile throughput contribution
                reward += (
                    -w_be_fifth
                    * (
                        (req_fifth_thr_normalized - slice_hist["fifth_perc_pkt_thr"])
                        / req_fifth_thr_normalized
                    )
                    if slice_hist["fifth_perc_pkt_thr"] < req_fifth_thr_normalized
                    else 0
                )

        return reward

    def update_ues_traffic(self) -> None:
        self.slice_requirements = {}
        for slice in self.slices:
            traffic_level = self.rng.integers(len(self.traffic_throughputs))
            is_be = slice.name == "be"
            be_prob = self.rng.random()
            self.slice_requirements[slice.name] = (
                {"long_term_pkt_thr": 0, "fifth_perc_pkt_thr": 0}
                if is_be and be_prob > 0.5
                else self.slice_requirements_traffics[
                    list(self.traffic_throughputs.keys())[traffic_level]
                ][slice.name]
            )

            for ue in slice.ues:
                ue.traffic_throughput = (
                    -1
                    if is_be and be_prob > 0.5
                    else self.traffic_throughputs[
                        list(self.traffic_throughputs.keys())[traffic_level]
                    ][ue.traffic_type]
                )

    @staticmethod
    def create_combinations(total_rbs: int, number_slices: int, full=False):
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
        path = ("{}/hist/{}/trial{}/").format(
            self.root_path, self.bs_name, self.trial_number
        )
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
    def read_hist(
        bs_name: str,
        trial_number: int,
        root_path: str = ".",
    ) -> tuple:
        """
        Read variables history from external file.
        """
        path = "{}/hist/{}/trial{}/bs.npz".format(root_path, bs_name, trial_number)
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
        root_path: str = ".",
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
                "se",
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
                "Spectral efficiency (bits/s/Hz)",
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
                    hist = Slice.read_hist(bs_name, trial_number, slice_id, root_path)[
                        plot_number
                    ]
                    plt.plot(
                        range(0, len(hist), step),
                        hist[0::step],
                        label="Slice {}".format(slices_name[slice_id - 1]),
                    )
                fig.tight_layout()
                plt.legend(fontsize=12)
                fig.savefig(
                    "{}/hist/{}/trial{}/{}.pdf".format(
                        root_path, bs_name, trial_number, filenames[plot_number]
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
                hist = Basestation.read_hist(bs_name, trial_number, root_path)[
                    plot_number
                ]
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
                    "{}/hist/{}/trial{}/{}.pdf".format(
                        root_path, bs_name, trial_number, filenames[plot_number]
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
    traffic_throughputs = {
        "light": {
            "embb": 15,
            "urllc": 1,
            "be": 5,
        },
        "moderate": {
            "embb": 25,
            "urllc": 5,
            "be": 10,
        },
    }
    slice_requirements_traffics = {
        "light": {
            "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
            "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
            "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
        },
        "moderate": {
            "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
            "urllc": {"throughput": 5, "latency": 1, "pkt_loss": 0.001},
            "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
        },
    }
    trials = 2
    rng = np.random.default_rng(1)
    basestation = Basestation(
        bs_name="test",
        max_number_trials=trials,
        traffic_types=traffic_types,
        traffic_throughputs=traffic_throughputs,
        slice_requirements_traffics=slice_requirements_traffics,
        obs_space_mode="partial",
        plots=True,
        save_hist=True,
        rng=rng,
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
