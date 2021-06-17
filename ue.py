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
        id: int,
        buffer_size: int,
        buffer_max_lat: int,
        bandwidth: float,
        packet_size: int,
        trial: int,
        traffic_type: str,
        frequency: int,
        total_number_rbs: int,
    ) -> None:
        self.id = id
        self.trial = trial
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.packet_size = packet_size
        self.traffic_type = traffic_type
        self.frequency = frequency
        self.total_number_rbs = total_number_rbs
        self.se = Channel.read_se_file(
            "./se/trial{}_f{}_ue{}.npy", trial, frequency, id
        )
        self.buffer = Buffer(buffer_size, buffer_max_lat)
        self.hist_labels = [
            "pkt_rcv",
            "pkt_snt",
            "pkt_thr",
            "buffer_occ",
            "avg_lat",
            "dropped_pkts",
        ]
        self.hist = {hist_label: np.array([]) for hist_label in self.hist_labels}

    def get_arrived_packets(self):
        """
        Return the number of packets received to queue in the buffer structure.
        It varies in according to the slice traffic behavior.
        """
        return 150  # returning constant number (TODO)

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
        dropped_packets: int,
    ) -> None:
        """
        Update the variables history to enable the record to external files.
        """
        hist_vars = [
            packets_received,
            packets_sent,
            packets_throughput,
            buffer_occupancy,
            avg_latency,
            dropped_packets,
        ]
        for i, var in enumerate(self.hist.items()):
            self.hist[var[0]] = np.append(self.hist[var[0]], hist_vars[i])

    def save_hist(self) -> None:
        """
        Save variables history to external file.
        """
        path = "./hist/trial{}/ues/"
        try:
            os.makedirs(path.format(self.trial))
        except OSError:
            pass

        np.savez_compressed((path + "ue{}").format(self.trial, self.id), **self.hist)
        UE.plot_metrics(self.trial, self.id)

    @staticmethod
    def read_hist(trial_number: int, ue_id: int) -> np.array:
        """
        Read variables history from external file.
        """
        path = "./hist/trial{}/ues/ue{}.npz".format(trial_number, ue_id)
        data = np.load(path)
        return np.array(
            [
                data.f.pkt_rcv,
                data.f.pkt_snt,
                data.f.pkt_thr,
                data.f.buffer_occ,
                data.f.avg_lat,
                data.f.dropped_pkts,
            ]
        )

    @staticmethod
    def plot_metrics(trial_number: int, ue_id: int) -> None:
        """
        Plot UE performance obtained over a specific trial. Read the
        information from external file.
        """
        hist = UE.read_hist(trial_number, ue_id)

        title_labels = [
            "Received Packets",
            "Sent Packets",
            "Packets Thr. Capacity",
            "Buffer Occupancy Rate",
            "Average Buffer Latency",
            "Dropped Buffer Packets",
        ]
        x_label = "Iteration [n]"
        y_labels = [
            "# pkts",
            "# pkts",
            "# pkts",
            "Occupancy rate",
            "Latency [ms]",
            "# pkts",
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
            "./hist/trial{}/ues/ue{}.png".format(trial_number, ue_id),
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
        )


def main():
    # Testing UE functions
    ue = UE(1, 1024, 10, 100, 2, 1, "embb", 1, 17)
    for i in range(2000):
        ue.step(i, 10)
    ue.save_hist()


if __name__ == "__main__":
    main()
