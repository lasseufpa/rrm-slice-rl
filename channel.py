import os
import re
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Channel:
    """
    Class containing the functions to calculate the SE for each UE in according
    to data obtained from QuaDriGa simulations.
    """

    @staticmethod
    def extract_power(
        path_to_rsrp_csv: str, no_cell: int, no_samples: int
    ) -> Tuple[np.array, np.array]:
        """
        Gets the linear power for the top 7 cells for a specific UEx_fc_y.csv.
        Output is power = [no_cell, no_samples], and serving_pci = [no_samples].
        Assumes that besides the 7 strongest cells, the remaining are effectively 0.
        """
        one_rsrp = pd.read_csv(path_to_rsrp_csv)
        powers = -1000 * np.ones(
            (no_cell, no_samples)
        )  # need a small starting dB value
        serving_index = one_rsrp["serving pci"] - 1
        for i in range(no_samples):
            powers[serving_index[i], i] = one_rsrp["serving rsrp"][i]

        # all powers are 0 now except for the serving ones. Now we do the same for each of the next 6, all else are effectively 0
        for j in range(6):
            # neigh_index = one_rsrp["neigh {} pci".format(j + 1)] - 1
            for i in range(no_samples):
                powers[serving_index[i], i] = one_rsrp["neigh {} rsrp".format(j + 1)][i]

        # now all of the primary power information for that fc-UE pair is available in powers
        powers = 10 ** (powers / 10)  # convert to linear
        return (powers, serving_index)

    @staticmethod
    def get_serving_se(
        sir_path: str,
        rsrp_path: str,
        no_fc: int,
        no_UE: int,
        no_cell: int,
        no_samples: int,
    ) -> np.array:
        """
        Calculate the spectral efficiency using Bjornson Massive MIMO Book EQ 7.1.
        Note that the current power allocation in the data is uniform and equal to 46dBm
        (split over the subcarriers, but the RSRP combines this back together).
        rho_set should correspond to rho_jk
        Also note! I have not added noise yet, so you will want to decide on a noise power
        """
        match_fc = ["2$", "28$"]
        match_ue = "UE{}_"
        sigma_sq = 1e-10
        # prelogFactor = 1
        rho_set = np.ones((no_fc, no_UE, no_cell))
        powers = np.zeros(
            (no_fc, no_UE, no_cell, no_samples)
        )  # store all the data here
        serving_indices = np.zeros(
            (no_fc, no_UE, no_samples), dtype=np.int16
        )  # useful for getting the best results
        if len(rho_set.shape) == 3:
            rho_set = np.tile(
                np.expand_dims(rho_set, -1), [1, 1, 1, no_samples]
            )  # expand and copy to the same shape as powers
        SIR_df = pd.read_csv(sir_path)

        # get the RSRP information -- this is basically Pt * abs(H)^2, i.e. optimal precoding
        for i, col in enumerate(SIR_df.columns):
            for j in range(no_UE):
                if re.search(match_ue.format(j + 1), col):
                    for k, fc in enumerate(match_fc):
                        if re.search(fc, col):
                            (
                                powers[k, j, :, :],
                                serving_indices[k, j, :],
                            ) = Channel.extract_power(
                                rsrp_path + col + ".csv", no_cell, no_samples
                            )

        # do something with rho_set and get SE
        SE = np.zeros((no_fc, no_UE, no_cell, no_samples))
        for i in range(no_fc):
            for j in range(no_UE):
                num = np.squeeze(rho_set[i, j, :, :]) * np.squeeze(
                    powers[i, j, :, :]
                )  # allows for any cell to be the serving cell
                den = (
                    np.sum(
                        np.squeeze(rho_set[i, j, :, :])
                        * np.squeeze(powers[i, j, :, :]),
                        0,
                    )
                    - num
                    + sigma_sq
                )
                SE[i, j, :, :] = num / den

        SE = np.log2(1 + SE)

        # Now, the best spectral efficiency for each UE and band is
        serving_SE = np.zeros((no_fc, no_UE, no_samples))
        for i in range(no_fc):
            for j in range(no_UE):
                for k in range(no_samples):
                    if i == 1:
                        serving_SE[i, j, k] = (
                            8 * SE[i, j, np.squeeze(serving_indices[i, j, k]), k]
                        )
                    else:
                        serving_SE[i, j, k] = SE[
                            i, j, np.squeeze(serving_indices[i, j, k]), k
                        ]

        return serving_SE

    @staticmethod
    def write_se_files(
        trials_list: list,
        file_path: str,
        sir_path: str,
        rsrp_path: str,
        no_fc: int,
        no_UE: int,
        no_cell: int,
        no_samples: int,
    ) -> None:
        """
        Write UE SE values to external files to save calculation time during the
        scenario simulation. Filenames follow the pattern
        './se/trial{number}_f{frequency_index}_ue{number}'.
        """
        try:
            os.mkdir("./se")
        except OSError as error:
            print(error)

        for trial in trials_list:
            serving_se = Channel.get_serving_se(
                sir_path.format(trial),
                rsrp_path.format(trial),
                no_fc,
                no_UE,
                no_cell,
                no_samples,
            )
            for frequency_index, se_frequency in enumerate(serving_se):
                for ue_index, se_ue in enumerate(se_frequency):
                    np.save(
                        file_path.format(trial, frequency_index + 1, ue_index + 1),
                        se_ue,
                    )

    @staticmethod
    def read_se_file(
        file_path: str,
        trial_number: int,
        frequency_index: int,
        ue_number: int,
        root_path: str = ".",
    ) -> np.array:
        """
        Read SE values for each UE from external files.
        """
        return np.load(
            file_path.format(root_path, trial_number, frequency_index, ue_number)
        )

    @staticmethod
    def plot_se(
        file_path: str,
        trial_number: int,
        frequency_index: int,
        ues_list: list,
        no_samples: int,
    ):
        """
        Plot SE values for a given list of UEs in a specific frequency and trial number.
        """
        labels = []
        plt.figure()
        for index, ue in enumerate(ues_list):
            plt.plot(
                np.arange(no_samples),
                Channel.read_se_file(file_path, trial_number, frequency_index, ue),
            )
            labels.append("UE {}".format(ue))
        plt.xlabel("time [ms]")
        plt.ylabel("SE [bps/Hz]")
        plt.grid()
        plt.title("{}GHz SE Comparison".format({1: "2", 2: "28"}[frequency_index]))
        plt.legend(labels)
        plt.show()


def main():
    # Write SE from trial 1 to external files
    sir_path = "channels/3gpp-UMi/trial {}/SIR_table.csv"
    rsrp_path = "channels/3gpp-UMi/trial {}/rsrp/"
    no_fc = 2
    no_UE = 10
    no_cell = 7 * 3  # 7 base stations, each with 3 sectors
    no_samples = 1000 * 2  # 1kHz sampling for 2 seconds
    file_path = "./se/trial{}_f{}_ue{}.npy"
    Channel.write_se_files(
        range(38, 51), file_path, sir_path, rsrp_path, no_fc, no_UE, no_cell, no_samples
    )

    # Plot SE from trial 1 frequency index 2 from external files
    # Channel.plot_se(file_path, 1, 1, np.arange(1, 11), no_samples)


if __name__ == "__main__":
    main()
