from itertools import product

import numpy as np


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
    ) -> None:
        if type == "mt":
            self.predict = self.max_throughput
        elif type == "rr":
            self.predict = self.round_robin
        elif type == "pf":
            self.predict = self.proportional_fair
        self.max_packets_buffer = max_packets_buffer
        self.total_rbs = total_rbs
        self.action_space = self.create_combinations(total_rbs, slices_number)
        self.round_robin_alloc = [6, 6, 5]
        self.tmp_throughput_sent = [1, 1, 1]

    def max_throughput(self, obs: np.array) -> int:
        throughputs_capacity = obs[[10, 18, 26]]
        buffer_pkt_occupancies = obs[[11, 18, 25]] * self.max_packets_buffer
        max_throughput_avail = np.minimum(throughputs_capacity, buffer_pkt_occupancies)
        perc_rbs_allocation = (
            (max_throughput_avail / np.sum(max_throughput_avail)) * self.total_rbs
            if np.sum(max_throughput_avail) != 0
            else [6, 6, 5]
        )
        action = np.argmin(
            np.sum(np.abs(self.action_space - perc_rbs_allocation), axis=1)
        )
        return action, []

    def round_robin(self, obs: np.array) -> int:
        self.round_robin_alloc = np.roll(self.round_robin_alloc, 1)
        action = np.argmin(
            np.sum(np.abs(self.action_space - self.round_robin_alloc), axis=1)
        )
        return action, []

    def proportional_fair(self, obs: np.array) -> int:
        throughputs_capacity = obs[[10, 18, 26]]
        buffer_pkt_occupancies = obs[[11, 18, 25]] * self.max_packets_buffer
        max_throughput_avail = np.minimum(throughputs_capacity, buffer_pkt_occupancies)
        prop_fair_calc = (
            (
                self.total_rbs
                * (max_throughput_avail / self.tmp_throughput_sent)
                / np.sum(max_throughput_avail / self.tmp_throughput_sent)
            )
            if np.sum(max_throughput_avail) != 0
            else [6, 6, 5]
        )
        action = np.argmin(np.sum(np.abs(self.action_space - prop_fair_calc), axis=1))
        self.tmp_throughput_sent = max_throughput_avail
        self.tmp_throughput_sent[self.tmp_throughput_sent == 0] = 1
        return action, []

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

    def set_env(self, _):
        pass

    def main(self):
        pass

    if __name__ == "__main__":
        main()
