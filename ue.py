import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

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
        max_packets_buffer: int,
        buffer_max_lat: int,
        bandwidth: float,
        packet_size: int,
        run_number: int,
        trial_number: int,
        traffic_type: str,
        frequency: int,
        total_number_rbs: int,
        traffic_throughput: float,
        plots: bool,
        seed: int,  # - 1 represents no fixed seed
        windows_size: int = 100,
    ) -> None:
        self.bs_name = bs_name
        self.id = id
        self.run_number = run_number
        self.trial_number = trial_number
        self.max_packets_buffer = max_packets_buffer
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.traffic_type = traffic_type
        self.frequency = frequency
        self.total_number_rbs = total_number_rbs
        self.se = Channel.read_se_file(
            "./se/trial{}_f{}_ue{}.npy", trial_number, frequency, id
        )
        self.buffer = Buffer(max_packets_buffer, buffer_max_lat)
        self.traffic_throughput = traffic_throughput
        self.windows_size = windows_size
        self.plots = plots
        self.get_arrived_packets = self.define_traffic_function()
        self.hist_labels = [
            "pkt_rcv",
            "pkt_snt",
            "pkt_thr",
            "buffer_occ",
            "avg_lat",
            "pkt_loss",
            "se",
        ]
        self.hist = {hist_label: np.array([]) for hist_label in self.hist_labels}
        self.rng = (
            np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
        )

    def define_traffic_function(self):
        """
        Return a function to calculate the number of packets received to queue
        in the buffer structure. It varies in according to the slice traffic behavior.
        """

        def traffic_embb():
            return np.floor(
                np.abs(
                    self.rng.normal(
                        (self.traffic_throughput * 1e6) / self.packet_size,
                        10,
                    )
                )
            )

        def traffic_urllc():
            return np.floor(
                np.abs(
                    self.rng.poisson((self.traffic_throughput * 1e6) / self.packet_size)
                )
            )

        def traffic_be():
            if self.rng.random() >= 0.5:
                return np.floor(
                    np.abs(
                        self.rng.normal(
                            (self.traffic_throughput * 1e6) / self.packet_size,
                            10,
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
        """
        if step_number < self.windows_size:
            slice_idx = 0
        elif step_number >= self.windows_size:
            slice_idx = -(self.windows_size - 1)

        pkt_loss_den = np.sum(
            np.append(self.hist["pkt_rcv"][slice_idx:], packets_received)
        ) + np.sum(self.buffer.buffer)
        hist_vars = [
            np.mean(np.append(self.hist["pkt_rcv"][slice_idx:], packets_received)),
            np.mean(np.append(self.hist["pkt_snt"][slice_idx:], packets_sent)),
            np.mean(np.append(self.hist["pkt_thr"][slice_idx:], packets_throughput)),
            np.mean(np.append(self.hist["buffer_occ"][slice_idx:], buffer_occupancy)),
            np.mean(np.append(self.hist["avg_lat"][slice_idx:], avg_latency)),
            np.sum(np.append(self.hist["pkt_loss"][slice_idx:], pkt_loss))
            / pkt_loss_den
            if pkt_loss_den != 0
            else 0,
            np.mean(np.append(self.hist["se"][slice_idx:], self.se[step_number])),
        ]
        for i, var in enumerate(self.hist.items()):
            self.hist[var[0]] = np.append(self.hist[var[0]], hist_vars[i])

    def save_hist(self) -> None:
        """
        Save variables history to external file.
        """
        path = "./hist/{}/run_{}/trial{}/ues/".format(
            self.bs_name, self.run_number, self.trial_number
        )
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed((path + "ue{}").format(self.id), **self.hist)
        if self.plots:
            UE.plot_metrics(self.bs_name, self.run_number, self.trial_number, self.id)

    @staticmethod
    def read_hist(
        bs_name: str, run_number: int, trial_number: int, ue_id: int
    ) -> np.array:
        """
        Read variables history from external file.
        """
        path = "./hist/{}/run_{}/trial{}/ues/ue{}.npz".format(
            bs_name, run_number, trial_number, ue_id
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
            ]
        )

    @staticmethod
    def plot_metrics(
        bs_name: str, run_number: int, trial_number: int, ue_id: int
    ) -> None:
        """
        Plot UE performance obtained over a specific trial. Read the
        information from external file.
        """
        hist = UE.read_hist(bs_name, run_number, trial_number, ue_id)

        title_labels = [
            "Received Packets",
            "Sent Packets",
            "Packets Thr. Capacity",
            "Buffer Occupancy Rate",
            "Average Buffer Latency",
            "Packet Loss Rate",
        ]
        x_label = "Iteration [n]"
        y_labels = [
            "# pkts",
            "# pkts",
            "# pkts",
            "Occupancy rate",
            "Latency [ms]",
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
            "./hist/{}/run_{}/trial{}/ues/ue{}.png".format(
                bs_name, run_number, trial_number, ue_id
            ),
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=100,
        )
        plt.close()

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
        max_packets_buffer=1024,
        buffer_max_lat=10,
        bandwidth=100,
        packet_size=2,
        run_number=1,
        trial_number=1,
        traffic_type="embb",
        frequency=1,
        total_number_rbs=17,
        traffic_throughput=10,
        plots=False,
        seed=2021,
        windows_size=100,
    )
    for i in range(2000):
        ue.step(i, 10)
    ue.save_hist()


if __name__ == "__main__":
    main()
