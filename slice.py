import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from ue import UE


class Slice:
    """
    Slice class containing the slice functions. Each slice has a list with UEs
    and it is responsible to allocate the RBs allocated to the slice to the UEs
    following a Round Robin algorithm. Each slice will be assigned to a base
    station.
    """

    def __init__(self, id: int, ues: list) -> None:
        self.id = id
        self.ues = ues
        self.hist_labels = [
            "pkt_rcv",
            "pkt_snt",
            "pkt_thr",
            "buffer_occ",
            "avg_lat",
            "dropped_pkts",
        ]
        self.hist = {hist_label: np.array([]) for hist_label in self.hist_labels}
        self.ues_order = []
        self.num_rbgs_assigned = 0

    def add_ue(self, ue: UE) -> None:
        """
        Add a UE to the slice UEs list.
        """
        self.ues = np.append(self.ues, ue)

    def assign_rbs_slice(self, num_rbs: int) -> None:
        """
        Assign the number of RBs specified by the base station to the slice.
        """
        self.num_rbgs_assigned = num_rbs

    def update_hist(self, hist_ues: list) -> None:
        """
        Update slice variables history to enable the record to external files.
        """
        hist_ue_labels = [
            "pkt_rcv",
            "pkt_snt",
            "pkt_thr",
            "buffer_occ",
            "avg_lat",
            "dropped_pkts",
        ]
        hist_vars = np.array([])
        for label in hist_ue_labels:
            hist_vars = np.append(
                hist_vars, np.mean([hist_ue[label][-1] for hist_ue in hist_ues])
            )

        for i, var in enumerate(self.hist.items()):
            self.hist[var[0]] = np.append(self.hist[var[0]], hist_vars[i])

    def get_last_hist(self) -> dict:
        """
        Return a hist variable containing the last iteration values.
        """
        return {
            hist_label: self.hist[hist_label][-1].item()
            for hist_label in self.hist_labels
        }

    def save_hist(self, trial_number: int) -> None:
        """
        Save slice variables history to external file.
        """
        path = "./hist/trial{}/slices/"
        try:
            os.makedirs(path.format(trial_number))
        except OSError:
            pass

        np.savez_compressed(
            (path + "slice{}").format(trial_number, self.id), **self.hist
        )
        Slice.plot_metrics(trial_number, self.id)

    @staticmethod
    def read_hist(trial_number: int, slice_id: int) -> None:
        """
        Read slice variables history from external file.
        """
        path = "./hist/trial{}/slices/slice{}.npz".format(trial_number, slice_id)
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
    def plot_metrics(trial_number: int, slice_id: int) -> None:
        """
        Plot slice performance obtained over a specific trial. Read the
        information from external file.
        """
        hist = Slice.read_hist(trial_number, slice_id)

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
        fig.suptitle("Trial {}, Slice {}".format(trial_number, slice_id))

        for i in np.arange(len(title_labels)):
            ax = fig.add_subplot(3, 2, i + 1)
            ax.set_title(title_labels[i])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_labels[i])
            ax.scatter(np.arange(hist[i].shape[0]), hist[i])
            ax.grid()
        fig.tight_layout()
        fig.savefig(
            "./hist/trial{}/slices/slice{}.png".format(trial_number, slice_id),
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=100,
        )
        plt.close()

    def step(
        self, step_number: int, max_step_number: int, num_rbs_allocated: int
    ) -> None:
        """
        Executes slice processing. It allocates the RBs received from the base
        station to the UEs following a round robin algorithm.
        """
        # Consider a round-robin allocation among UEs
        rbs_ues = np.zeros(len(self.ues))
        pool = cycle(np.arange(len(rbs_ues))[::-1])
        for i in np.arange(num_rbs_allocated):
            rbs_ues[next(pool)] += 1

        # Allocating assigned RBs to UEs
        hist_ues = []
        for i, ue in enumerate(self.ues):
            ue.step(step_number, rbs_ues[i])
            hist_ues.append(ue.hist)
            if step_number == (max_step_number - 1):
                ue.save_hist()

        #  Rools UE array to enable round-robin allocation
        self.ues = np.roll(self.ues, num_rbs_allocated)

        # Update slice history
        self.update_hist(hist_ues)


def main():
    const_rbs = 2
    number_ues = 3
    max_number_steps = 2000
    ues = [
        UE(i, 1024, 10, 100, 2, 1, "embb", 1, 17) for i in np.arange(1, number_ues + 1)
    ]
    slice = Slice(1, ues)
    for i in range(max_number_steps):
        slice.step(i, max_number_steps, const_rbs)
    slice.save_hist(1)


if __name__ == "__main__":
    main()
