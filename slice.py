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

    def __init__(
        self,
        bs_name: str,
        id: int,
        name: str,
        trial_number: int,
        ues: list,
        plots: bool,
        save_hist: bool = False,
        root_path: str = ".",
    ) -> None:
        self.bs_name = bs_name
        self.id = id
        self.name = name
        self.trial_number = trial_number
        self.ues = ues
        self.plots = plots
        self.save_hist_bool = save_hist
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
        self.ues_order = []
        self.num_rbgs_assigned = 0
        self.rr_index = 0
        self.root_path = root_path

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

    def update_hist(self, hist_ues: list, hist_nowindows_ues: list) -> None:
        """
        Update slice variables history to enable the record to external files.
        """
        hist_ue_labels = [
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
        hist_vars = np.array([])
        hist_nowindows_vars = np.array([])
        for label in hist_ue_labels:
            hist_vars = np.append(
                hist_vars, np.mean([hist_ue[label][-1] for hist_ue in hist_ues])
            )
            hist_nowindows_vars = np.append(
                hist_nowindows_vars,
                np.mean(
                    [
                        hist_nowindows_ue[label][-1]
                        for hist_nowindows_ue in hist_nowindows_ues
                    ]
                ),
            )

        for i, var in enumerate(self.hist.items()):
            self.hist[var[0]] = np.append(self.hist[var[0]], hist_vars[i])
            self.no_windows_hist[var[0]] = np.append(
                self.no_windows_hist[var[0]], hist_nowindows_vars[i]
            )

    def get_last_no_windows_hist(self) -> dict:
        """
        Return a hist variable containing the last iteration values.
        """
        return {
            hist_label: self.no_windows_hist[hist_label][-1].item()
            for hist_label in self.hist_labels
        }

    def save_hist(self) -> None:
        """
        Save slice variables history to external file.
        """
        path = "{}/hist/{}/trial{}/slices/".format(
            self.root_path, self.bs_name, self.trial_number
        )
        try:
            os.makedirs(path)
        except OSError:
            pass

        np.savez_compressed((path + "slice{}").format(self.id), **self.no_windows_hist)
        if self.plots:
            Slice.plot_metrics(self.bs_name, self.trial_number, self.id, self.root_path)

    @staticmethod
    def read_hist(
        bs_name: str, trial_number: int, slice_id: int, root_path: str = "."
    ) -> None:
        """
        Read slice variables history from external file.
        """
        path = "{}/hist/{}/trial{}/slices/slice{}.npz".format(
            root_path, bs_name, trial_number, slice_id
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
        bs_name: str, trial_number: int, slice_id: int, root_path: str = "."
    ) -> None:
        """
        Plot slice performance obtained over a specific trial. Read the
        information from external file.
        """
        hist = Slice.read_hist(bs_name, trial_number, slice_id, root_path)

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
            "{}/hist/{}/trial{}/slices/slice{}.png".format(
                root_path, bs_name, trial_number, slice_id
            ),
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=100,
        )
        plt.close()

    def step(
        self,
        step_number: int,
        max_step_number: int,
        num_rbs_allocated: int,
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

        rbs_ues = np.roll(rbs_ues, self.rr_index)
        self.rr_index += 1 if self.rr_index < (len(self.ues) - 1) else -self.rr_index

        # Allocating assigned RBs to UEs
        hist_ues = []
        hist_nowindows_ues = []
        for i, ue in enumerate(self.ues):
            ue.step(step_number, rbs_ues[i])
            hist_ues.append(ue.hist)
            hist_nowindows_ues.append(ue.no_windows_hist)
            if (step_number == (max_step_number - 1)) and self.save_hist_bool:
                ue.save_hist()

        # Update slice history
        self.update_hist(hist_ues, hist_nowindows_ues)


def main():
    const_rbs = 2
    number_ues = 3
    max_number_steps = 2000
    rng = np.random.default_rng(1)
    ues = [
        UE(
            bs_name="test",
            id=i,
            trial_number=1,
            traffic_type="embb",
            traffic_throughput=50,
            plots=False,
            rng=rng,
        )
        for i in np.arange(1, number_ues + 1)
    ]
    slice = Slice(
        bs_name="test",
        id=1,
        name="slice_name",
        trial_number=1,
        ues=ues,
        plots=False,
    )
    for i in range(max_number_steps):
        slice.step(i, max_number_steps, const_rbs)
    slice.save_hist()


if __name__ == "__main__":
    main()
