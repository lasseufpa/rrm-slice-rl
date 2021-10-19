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

    def max_throughput(self, obs: np.array) -> int:
        throughputs_capacity = (
            obs[[10, 18, 26]] * self.bandwidth * (1 / self.total_number_rbs)
        ) / self.packet_size
        buffer_pkt_occupancies = obs[[11, 20, 29]] * self.max_packets_buffer
        max_pkt_throughput_avail = np.minimum(
            throughputs_capacity, buffer_pkt_occupancies
        )
        action = (
            (max_pkt_throughput_avail / np.sum(max_pkt_throughput_avail))
            if np.sum(max_pkt_throughput_avail) != 0
            else [1, 1, 1]
        )
        return np.array(action), []

    def round_robin(self, obs: np.array) -> int:
        self.round_robin_alloc = np.roll(self.round_robin_alloc, 1)
        action = self.round_robin_alloc / np.sum(self.round_robin_alloc)
        return np.array(action), []

    def proportional_fair(self, obs: np.array) -> int:
        throughputs_capacity = (
            obs[[10, 18, 26]] * self.bandwidth * (1 / self.total_number_rbs)
        ) / self.packet_size
        buffer_pkt_occupancies = obs[[11, 20, 29]] * self.max_packets_buffer
        max_pkt_throughput_avail = np.minimum(
            throughputs_capacity, buffer_pkt_occupancies
        )
        snt_pkt_throughput = obs[[9, 18, 27]]
        snt_pkt_throughput[snt_pkt_throughput == 0] = 0.00001
        fairness_calc = max_pkt_throughput_avail / snt_pkt_throughput
        action = (
            (self.total_rbs * fairness_calc / np.sum(fairness_calc))
            if np.sum(fairness_calc) != 0
            else [1, 1, 1]
        )
        return np.array(action), []

    def set_env(self, _):
        pass

    def main(self):
        pass

    if __name__ == "__main__":
        main()
