from itertools import product

import numpy as np

from basestation import Basestation


class BaselineAgent:
    """
    Class containing the UE functions. Each UE have a buffer and Channel values
    for specific trials. Each UE will be assigned to a slice.
    """

    def __init__(
        self,
        type: str,
        max_packets_buffer: int = 1024,
        total_rbs: int = 17,
        slices_number: int = 3,
        bandwidth: float = 100000000,
        total_number_rbs: int = 17,
        packet_size: int = 8192 * 8,
    ) -> None:
        if type == "mt":
            self.predict = self.max_throughput
        elif type == "rr":
            self.predict = self.round_robin
        elif type == "pf":
            self.predict = self.proportional_fair
        self.max_packets_buffer = max_packets_buffer
        self.total_rbs = total_rbs
        self.bandwidth = bandwidth
        self.total_number_rbs = total_number_rbs
        self.packet_size = packet_size
        self.action_space = Basestation.create_combinations(
            total_rbs, slices_number, full=True
        )
        self.round_robin_alloc = [6, 6, 5]
        self.vec_throughput_snt = np.empty((0, 3))

    def max_throughput(self, obs: np.array) -> int:
        se = obs[[14, 23, 32]]
        buffer_occ = obs[[11, 20, 29]]
        total_throughput_avail = np.min(
            [
                se * self.bandwidth,
                buffer_occ * self.max_packets_buffer * self.packet_size,
            ],
            axis=0,
        )

        return np.array(total_throughput_avail), []

    def round_robin(self, obs: np.array) -> int:
        self.round_robin_alloc = np.roll(self.round_robin_alloc, 1)

        return np.array(self.round_robin_alloc), []

    def proportional_fair(self, obs: np.array) -> int:
        self.vec_throughput_snt = np.append(
            self.vec_throughput_snt, [obs[[9, 18, 27]]], axis=0
        )
        se = obs[[14, 23, 32]]
        buffer_occ = obs[[11, 20, 29]]
        total_throughput_avail = np.min(
            [
                se * self.bandwidth,
                buffer_occ * self.max_packets_buffer * self.packet_size,
            ],
            axis=0,
        )
        factors = (
            total_throughput_avail / np.mean(self.vec_throughput_snt, axis=0)
            if not (0 in np.mean(self.vec_throughput_snt, axis=0))
            else [1, 1, 1]
        )

        return np.array(factors), []

    def set_env(self, _):
        pass

    def main(self):
        pass

    def set_random_seed(self, seed):
        pass

    if __name__ == "__main__":
        main()
