import os

import matplotlib.pyplot as plt
import numpy as np

from basestation import Basestation
from slice import Slice

slices = {
    "be": 1,
    "embb": 2,
    "urllc": 3,
}
slices_names_markers = {
    "be": ("BE", "*"),
    "embb": ("eMBB", "p"),
    "urllc": ("URLLC", "d"),
}
agents_names_colors = {
    "sac": ("SAC", "#003f5c"),
    "intentless": ("Lower-level [11]", "#444e86"),
    "colran": ("Sched-slicing [9]", "#955196"),
    "rr": ("RR", "#dd5182"),
    "pf": ("PF", "#ff6e54"),
    "mt": ("MT", "#ffa600"),
}
data_index = {
    "throughput": (2, "Served throughput (Mbps)"),
    "latency": (4, "Latency (ms)"),
    "pkt_loss": (5, "Packet loss rate"),
    "long_term_pkt_thr": (7, "Long-term throughput (Mbps)"),
    "fifth_perc_pkt_thr": (8, "Fifth-percentile throughput (Mbps)"),
}
ws_names_colors = {
    1: "#003f5c",
    50: "#444e86",
    100: "#955196",
}
obs_names_colors = {
    "full": ("Full", "#003f5c"),
    "partial": ("Limited", "#444e86"),
}
obs_names_markers = {
    "full": ("Full", "*"),
    "partial": ("Limited", "p"),
}
ws_markers = {
    1: "*",
    10: "o",
    50: "p",
    100: "d",
}


def plot_agents_reqs(
    fig_name: str,
    trial_numbers: list,
    slices_req: dict,
    agents: list,
    windows_sizes: list,
    obs_spaces: list,
) -> None:
    x_label = "Time (ms)"

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(data_index[attribute][1], fontsize=14)
        plt.grid()
        for slice in slices.keys():
            label_slices = (
                "{}".format(slices_names_markers[slice][0]) if len(slices) > 1 else ""
            )
            for obs_space in obs_spaces:
                label_obs_space = (
                    ", {} Obs. Space".format(obs_space) if len(obs_spaces) > 1 else ""
                )
                for windows_size in windows_sizes:
                    label_windows_size = (
                        ", $W_s={}$".format(windows_size) if len(obs_spaces) > 1 else ""
                    )
                    for agent in agents:
                        label_agent = (
                            ", {}".format(agents_names_colors[agent][0])
                            if len(agents) > 1
                            else ""
                        )
                        if (
                            attribute
                            in slices_req[list(slices_req.keys())[0]][slice].keys()
                        ):
                            # req_values = [0, slices_req]
                            hist = np.array([])
                            for trial_number in trial_numbers:
                                hist = np.append(
                                    hist,
                                    Slice.read_hist(
                                        "test/{}/ws_{}/{}/".format(
                                            agent,
                                            windows_size,
                                            obs_space,
                                        ),
                                        trial_number,
                                        slices[slice],
                                    )[data_index[attribute][0]],
                                )
                            x_values = range(0, len(hist))
                            markevery = 200
                            if attribute == "throughput":
                                x_values = np.sort(hist)
                                hist = 1.0 * np.arange(len(hist)) / (len(hist) - 1)
                                x_values = np.append(
                                    x_values,
                                    np.arange(np.max(x_values) + 1, 70 + 1),
                                )
                                hist = np.append(
                                    hist, np.ones(len(x_values) - len(hist))
                                )
                                markevery = 100
                                plt.xlabel("Served throughput (Mbps)", fontsize=14)
                                plt.ylabel(
                                    "Cumulative distribution function (CDF)",
                                    fontsize=14,
                                )
                            if (
                                attribute == "latency"
                                and agent == "intentless"
                                and slice == "embb"
                            ):
                                pass
                            else:
                                plt.plot(
                                    x_values,
                                    hist,
                                    label=label_slices
                                    + label_agent
                                    + label_windows_size
                                    + label_obs_space,
                                    markerfacecolor="None",
                                    marker=ws_markers[windows_size]
                                    if len(windows_sizes) > 1
                                    else obs_names_markers[obs_space][1]
                                    if len(obs_spaces) > 1
                                    else slices_names_markers[slice][1]
                                    if len(slices) > 1
                                    else None,
                                    color=agents_names_colors[agent][1],
                                    markevery=markevery,
                                )
                    # if attribute in slices_req[traffic][slice].keys():
                    #     plt.plot(
                    #         range(steps_number),
                    #         slices_req[traffic][slice][attribute] * np.ones(len(hist)),
                    #         linestyle="--",
                    #         marker=slices_names_markers[slice][1],
                    #         color="blue",
                    #         markevery=200,
                    #         zorder=3,
                    #         label="{} Req.".format(slices_names_markers[slice][0]),
                    #     )
        fig.tight_layout()
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/{}_{}.pdf".format(fig_name, attribute),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_rewards(
    fig_name: str,
    trial_numbers: list,
    agents: list,
    windows_sizes: list,
    obs_spaces: list,
    order: list = [],
    cumulative: bool = False,
) -> None:
    x_label = "Time (ms)"

    data_index = {
        "reward": 1,
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Cumulative reward", fontsize=14)
        plt.grid()
        for obs_space in obs_spaces:
            label_obs_space = (
                ", {} Obs. Space".format(obs_names_markers[obs_space][0])
                if len(obs_spaces) > 1
                else ""
            )
            for windows_size in windows_sizes:
                label_windows_size = (
                    ", $W_s={}$".format(windows_size) if len(windows_sizes) > 1 else ""
                )
                for agent in agents:
                    if agent in ["rr", "pf", "mt"] and obs_space == "partial":
                        pass
                    else:
                        label_agent = (
                            "{}".format(agents_names_colors[agent][0])
                            if len(agents) > 1
                            else ""
                        )
                        hist = np.array([])
                        for trial_number in trial_numbers:
                            hist = np.append(
                                hist,
                                Basestation.read_hist(
                                    "test/{}/ws_{}/{}/".format(
                                        agent,
                                        windows_size,
                                        obs_space,
                                    ),
                                    trial_number,
                                )[data_index[attribute]],
                            )
                        hist = np.cumsum(hist) if cumulative else hist
                        plt.plot(
                            range(0, len(hist)),
                            hist,
                            label=label_agent + label_windows_size + label_obs_space,
                            markerfacecolor="None",
                            marker=ws_markers[windows_size]
                            if len(windows_sizes) > 1
                            else obs_names_markers[obs_space][1]
                            if len(obs_spaces) > 1
                            else None,
                            color=agents_names_colors[agent][1]
                            if len(agents) > 1
                            else obs_names_colors[obs_space][1]
                            if len(obs_spaces) > 1
                            else ws_names_colors[windows_size],
                            markevery=200,
                        )
        fig.tight_layout()
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        if any(order):
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/{}.pdf".format(fig_name),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_rcv_thr(
    fig_name: str,
    trial_numbers: list,
) -> None:
    x_label = "Time (ms)"

    w, h = plt.figaspect(0.6)
    fig = plt.figure(figsize=(w, h))
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Requested throughput (Mbps)", fontsize=14)
    plt.grid()
    color = {
        "be": "#ff6e54",
        "embb": "#444e86",
        "urllc": "#955196",
    }
    for slice in slices.keys():
        label_slices = (
            "{}".format(slices_names_markers[slice][0]) if len(slices) > 1 else ""
        )
        agent = "sac"
        windows_size = 1
        obs_space = "full"
        # req_values = [0, slices_req]
        hist = np.array([])
        for trial_number in trial_numbers:
            hist = np.append(
                hist,
                Slice.read_hist(
                    "test/{}/ws_{}/{}/".format(
                        agent,
                        windows_size,
                        obs_space,
                    ),
                    trial_number,
                    slices[slice],
                )[0],
            )
        plt.plot(
            range(0, len(hist)),
            hist,
            label=label_slices,
            color=color[slice],
            # markevery=200,
            # markerfacecolor="None",
            # marker=slices_names_markers[slice][1],
        )

    fig.tight_layout()
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    os.makedirs("./results", exist_ok=True)
    fig.savefig(
        "./results/{}_{}.pdf".format(fig_name, "rcv_thr"),
        # bbox_inches="tight",
        pad_inches=0,
        format="pdf",
        dpi=1000,
    )
    # plt.show()
    plt.close()


trial_number = 50
agents = ["sac", "td3", "rr", "mt"]
windows_sizes = 10
observations_spaces = "partial"
traffics = "light"
runs = 10
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

##### Comparing different windows sizes for full and partial obs space
plot_rewards(
    "reward",
    [46, 47, 48, 49, 50],
    # ["td3"],
    ["pf", "sac", "intentless", "colran"],
    # [1, 50, 100],
    [1],
    ["full", "partial"],
    # [2, 0, 1],
    cumulative=True,
)

plot_agents_reqs(
    "metrics",
    [46],
    slice_requirements_traffics,
    ["pf", "sac", "intentless", "colran"],
    [1],
    ["partial"],
)

plot_rcv_thr("requested", [46])  # np.arange(46, 51),
