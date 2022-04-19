import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import BitGenerator

from buffer import Buffer
from channel import Channel


class UE:
    """
    Class containing the UE functions. Each UE have a buffer and Channel values
    for specific trials. Each UE will be assigned to a slice.
    """

    def __init__(
        self,
        bs_name: str,
        id: int,
        trial_number: int,
        traffic_type: str,
        traffic_throughput: float,
        max_packets_buffer: int = 1024,
        buffer_max_lat: int = 100,
        bandwidth: float = 100000000,
        packet_size: int = 8192 * 8,
        frequency: int = 2,
        total_number_rbs: int = 17,
        plots: bool = False,
        rng: BitGenerator = np.random.default_rng(),
        windows_size_obs: int = 1,
        windows_size: int = 10,
        normalize_obs: bool = False,
        root_path: str = ".",
    ) -> None:
        self.bs_name = bs_name
        self.id = id
        self.trial_number = trial_number
        self.max_packets_buffer = max_packets_buffer
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.traffic_type = traffic_type
        self.frequency = frequency
        self.total_number_rbs = total_number_rbs
        self.root_path = root_path
        self.se = Channel.read_se_file(
            "{}/se/trial{}_f{}_ue{}.npy", trial_number, frequency, id, self.root_path
        )
        self.buffer_max_lat = buffer_max_lat
        self.buffer = Buffer(max_packets_buffer, buffer_max_lat)
        self.traffic_throughput = traffic_throughput
        self.windows_size_obs = windows_size_obs
        self.windows_size = windows_size
        self.plots = plots
        self.normalize_obs = normalize_obs
        self.get_arrived_packets = self.define_traffic_function()
        self.hist_labels = [
            "pkt_rcv",
            "pkt_snt",
            "pkt_thr",
            "buffer_occ",
            "avg_lat",
            "pkt_loss",
            "se",
            "long_term_pkt_thr",
            "fifth_perc_pkt_thr",
        ]
        self.hist = {hist_label: np.array([]) for hist_label in self.hist_labels}
        self.no_windows_hist = {
            hist_label: np.array([]) for hist_label in self.hist_labels
        }
        self.number_pkt_loss = np.array([])
        self.rng = rng

    def define_traffic_function(self):
        """
        Return a function to calculate the number of packets received to queue
        in the buffer structure. It varies in according to the slice traffic behavior.
        """

        def traffic_embb():
            return np.floor(
                np.abs(
                    self.rng.poisson((self.traffic_throughput * 1e6) / self.packet_size)
                )
            )

        def traffic_urllc():
            return np.floor(
                np.abs(
                    self.rng.poisson((self.traffic_throughput * 1e6) / self.packet_size)
                )
            )

        def traffic_be():
            if self.traffic_throughput != -1:
                return np.floor(
                    np.abs(
                        self.rng.poisson(
                            (self.traffic_throughput * 1e6) / self.packet_size
                        )
                    )
                )
            else:
                return 0

        if self.traffic_type == "embb":
            return traffic_embb
        elif self.traffic_type == "urllc":
            return traffic_urllc
        elif self.traffic_type == "be":
            return traffic_be
        else:
            raise Exception(
                "UE {} traffic type {} specified is not valid".format(
                    self.id, self.traffic_type
                )
            )

    def get_pkt_throughput(
        self, step_number: int, number_rbs_allocated: int
    ) -> np.array:
        """
        Calculate the throughput available to be sent by the UE given the number
        of RBs allocated, bandwidth and the spectral efficiency. It is not the
        real throughput since the UE may have less packets in the buffer than
        the number of packets available to send.
        """
        return np.floor(
            (
                (number_rbs_allocated / self.total_number_rbs)
                * self.bandwidth
                * self.se[step_number]
            )
            / self.packet_size
        )

    def update_hist(
        self,
        packets_received: int,
        packets_sent: int,
        packets_throughput: int,
        buffer_occupancy: float,
        avg_latency: float,
        pkt_loss: int,
        step_number: int,
    ) -> None:
        """
        Update the variables history to enable the record to external files.
        It creates a tmp_hist that records the history of UE variables without
        using a windows calculation and it is used as basis to calculate the
        hist variable using windows average.
        """
        hist_vars = [
            self.packets_to_mbps(self.packet_size, packets_received),
            self.packets_to_mbps(self.packet_size, packets_sent),
            self.packets_to_mbps(self.packet_size, packets_throughput),
            buffer_occupancy,
            avg_latency,
            pkt_loss,
            self.se[step_number],
        ]
        normalize_factors = (
            [100, 100, 100, 1, self.buffer_max_lat, 1, 100, 100, 100]
            if self.normalize_obs
            else np.ones(len(self.hist.keys()))
        )
        self.number_pkt_loss = np.append(self.number_pkt_loss, pkt_loss)

        # Hist with no windows for log (not used in the observation space)
        idx = (
            slice(-(self.windows_size - 1), None)
            if self.windows_size != 1
            else slice(0, 0)
        )
        for i, var in enumerate(self.no_windows_hist.items()):
            if var[0] == "pkt_loss":
                buffer_pkts = (
                    np.sum(self.buffer.buffer)
                    + np.sum(self.no_windows_hist["pkt_snt"][idx])
                    + np.sum(self.number_pkt_loss[idx])
                    - np.sum(self.no_windows_hist["pkt_rcv"][idx])
                )
                den = (
                    np.sum(self.no_windows_hist["pkt_rcv"][idx])
                    + hist_vars[0]
                    + buffer_pkts
                )
                self.no_windows_hist[var[0]] = (
                    np.append(
                        self.no_windows_hist[var[0]],
                        (np.sum(self.number_pkt_loss[idx]) + hist_vars[i]) / den,
                    )
                    if den != 0
                    else np.append(self.no_windows_hist[var[0]], 0)
                )
            elif var[0] == "long_term_pkt_thr":
                self.no_windows_hist[var[0]] = np.append(
                    self.no_windows_hist[var[0]],
                    np.sum(self.no_windows_hist["pkt_thr"][-self.windows_size :])
                    / self.no_windows_hist["pkt_thr"][-self.windows_size :].shape[0],
                )
            elif var[0] == "fifth_perc_pkt_thr":
                self.no_windows_hist[var[0]] = np.append(
                    self.no_windows_hist[var[0]],
                    np.percentile(
                        self.no_windows_hist["pkt_thr"][-self.windows_size :], 5
                    ),
                )
            else:
                self.no_windows_hist[var[0]] = np.append(
                    self.no_windows_hist[var[0]], hist_vars[i]
                )

        # Hist calculation to be used as observation space (using windows and normalization if applied)
        idx_obs = slice(-(self.windows_size_obs), None)
        for i, var in enumerate(self.hist.items()):
            value = (
                np.mean(self.no_windows_hist[var[0]][idx_obs])
                if self.no_windows_hist[var[0]].shape[0] != 0
                else 0
            )
            self.hist[var[0]] = np.append(
                self.hist[var[0]], value / normalize_factors[i]
            )

    def save_hist(self) -> None:
        """
        Save variables history to external file.
        """
        path = "{}/hist/{}/trial{}/ues/".format(
            self.root_path, self.bs_name, self.trial_number
        )
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed((path + "ue{}").format(self.id), **self.no_windows_hist)
        if self.plots:
            UE.plot_metrics(self.bs_name, self.trial_number, self.id, self.root_path)

    @staticmethod
    def read_hist(
        bs_name: str, trial_number: int, ue_id: int, root_path: str = "."
    ) -> np.array:
        """
        Read variables history from external file.
        """
        path = "{}/hist/{}/trial{}/ues/ue{}.npz".format(
            root_path, bs_name, trial_number, ue_id
        )
        data = np.load(path)
        return np.array(
            [
                data.f.pkt_rcv,
                data.f.pkt_snt,
                data.f.pkt_thr,
                data.f.buffer_occ,
                data.f.avg_lat,
                data.f.pkt_loss,
                data.f.se,
                data.f.long_term_pkt_thr,
                data.f.fifth_perc_pkt_thr,
            ]
        )

    @staticmethod
    def plot_metrics(
        bs_name: str, trial_number: int, ue_id: int, root_path: str = "."
    ) -> None:
        """
        Plot UE performance obtained over a specific trial. Read the
        information from external file.
        """
        hist = UE.read_hist(bs_name, trial_number, ue_id, root_path)

        title_labels = [
            "Received Throughput",
            "Sent Throughput",
            "Throughput Capacity",
            "Buffer Occupancy Rate",
            "Average Buffer Latency",
            "Packet Loss Rate",
        ]
        x_label = "Iteration [n]"
        y_labels = [
            "Throughput (Mbps)",
            "Throughput (Mbps)",
            "Throughput (Mbps)",
            "Occupancy rate",
            "Latency (ms)",
            "Packet loss rate",
        ]
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        fig.suptitle("Trial {}, UE {}".format(trial_number, ue_id))

        for i in np.arange(len(title_labels)):
            ax = fig.add_subplot(3, 2, i + 1)
            ax.set_title(title_labels[i])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_labels[i])
            ax.scatter(np.arange(hist[i].shape[0]), hist[i])
            ax.grid()
        fig.tight_layout()
        fig.savefig(
            "{}/hist/{}/trial{}/ues/ue{}.png".format(
                root_path, bs_name, trial_number, ue_id
            ),
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=100,
        )
        plt.close()

    @staticmethod
    def packets_to_mbps(packet_size, number_packets):
        return packet_size * number_packets / 1e6

    def step(self, step_number: int, number_rbs_allocated: int) -> None:
        """
        Executes the UE packets processing. Adding the received packets to the
        buffer and sending them in according to the throughput available and
        buffer.
        """
        pkt_throughput = self.get_pkt_throughput(step_number, number_rbs_allocated)
        pkt_received = self.get_arrived_packets()
        self.buffer.receive_packets(pkt_received)
        self.buffer.send_packets(pkt_throughput)
        self.update_hist(
            pkt_received,
            self.buffer.sent_packets,
            pkt_throughput,
            self.buffer.get_buffer_occupancy(),
            self.buffer.get_avg_delay(),
            self.buffer.dropped_packets,
            step_number,
        )


def main():
    # Testing UE functions
    ue = UE(
        bs_name="test",
        id=1,
        trial_number=1,
        traffic_type="embb",
        traffic_throughput=50,
        plots=False,
    )
    for i in range(2000):
        ue.step(i, 2)
    ue.save_hist()


if __name__ == "__main__":
    main()
