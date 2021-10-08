import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from basestation import Basestation
from slice import Slice


def plot_agents_comparison(
    trial_number: int,
    agents: list,
    slices_req: dict,
    windows_size_obs: int,
    traffic: str,
    obs_space: str,
    runs: int,
    steps_number: int = 2000,
) -> None:
    x_label = "Time (ms)"
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
        "a2c": ("A2C", "#003f5c"),
        "ppo": ("PPO", "#444e86"),
        "dqn": ("DQN", "#955196"),
        "rr": ("RR", "#dd5182"),
        "pf": ("PF", "#ff6e54"),
        "mt": ("MT", "#ffa600"),
    }
    data_index = {
        "throughput": (2, "Throughput (Mbps)"),
        "latency": (4, "Latency (ms)"),
        "pkt_loss": (5, "Packet loss rate"),
        "long_term_pkt_thr": (7, "Throughput (Mbps)"),
        "fifth_perc_pkt_thr": (8, "Throughput (Mbps)"),
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(data_index[attribute][1], fontsize=14)
        plt.grid()
        for slice in slices.keys():
            for agent in agents:
                if attribute in slice_requirements[traffic][slice].keys():
                    hist = np.zeros((runs, steps_number))
                    for run_number in range(runs):
                        hist[run_number, :] = Slice.read_hist(
                            "test/{}/ws_{}/{}/{}/run{}".format(
                                agent,
                                windows_size_obs,
                                obs_space,
                                traffic,
                                run_number + 1,
                            ),
                            trial_number,
                            slices[slice],
                        )[data_index[attribute][0]]
                    hist = np.mean(hist, axis=0)
                    hist = (
                        Basestation.packets_to_mbps(8192 * 8, hist)
                        if data_index[attribute][0] in [2, 7, 8]
                        else hist
                    )
                    plt.plot(
                        range(0, len(hist)),
                        hist,
                        label="{}, {}".format(
                            slices_names_markers[slice][0],
                            agents_names_colors[agent][0],
                        ),
                        marker=slices_names_markers[slice][1],
                        color=agents_names_colors[agent][1],
                        markevery=50,
                    )
            if attribute in slice_requirements[traffic][slice].keys():
                plt.plot(
                    range(steps_number),
                    slices_req[traffic][slice][attribute] * np.ones(steps_number),
                    linestyle="--",
                    marker=slices_names_markers[slice][1],
                    color="blue",
                    markevery=50,
                    zorder=3,
                    label="{} Req.".format(slices_names_markers[slice][0]),
                )
        fig.tight_layout()
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/agent_comp_{}_{}_{}_ws{}.pdf".format(
                attribute, traffic, obs_space, windows_size_obs
            ),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_ws_comparison(
    trial_number: int,
    agent: str,
    slices_req: dict,
    windows_sizes: list,
    traffic: str,
    obs_space: str,
    runs: int,
    steps_number: int = 2000,
) -> None:
    x_label = "Time (ms)"
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
    ws_names_colors = {
        1: "#003f5c",
        50: "#444e86",
        100: "#955196",
    }
    data_index = {
        "throughput": (2, "Throughput (Mbps)"),
        "latency": (4, "Latency (ms)"),
        "pkt_loss": (5, "Packet loss rate"),
        "long_term_pkt_thr": (7, "Throughput (Mbps)"),
        "fifth_perc_pkt_thr": (8, "Throughput (Mbps)"),
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(data_index[attribute][1], fontsize=14)
        plt.grid()
        for slice in slices.keys():
            for windows_size_obs in windows_sizes:
                if attribute in slice_requirements[traffic][slice].keys():
                    hist = np.zeros((runs, steps_number))
                    for run_number in range(runs):
                        hist[run_number, :] = Slice.read_hist(
                            "test/{}/ws_{}/{}/{}/run{}".format(
                                agent,
                                windows_size_obs,
                                obs_space,
                                traffic,
                                run_number + 1,
                            ),
                            trial_number,
                            slices[slice],
                        )[data_index[attribute][0]]
                    hist = np.mean(hist, axis=0)
                    hist = (
                        Basestation.packets_to_mbps(8192 * 8, hist)
                        if data_index[attribute][0] in [2, 7, 8]
                        else hist
                    )
                    plt.plot(
                        range(0, len(hist)),
                        hist,
                        label="{}, $W_s={}$".format(
                            slices_names_markers[slice][0],
                            windows_size_obs,
                        ),
                        marker=slices_names_markers[slice][1],
                        color=ws_names_colors[windows_size_obs],
                        markevery=50,
                    )
            if attribute in slice_requirements[traffic][slice].keys():
                plt.plot(
                    range(steps_number),
                    slices_req[traffic][slice][attribute] * np.ones(steps_number),
                    linestyle="--",
                    marker=slices_names_markers[slice][1],
                    color="blue",
                    markevery=50,
                    zorder=3,
                    label="{} Req.".format(slices_names_markers[slice][0]),
                )
        fig.tight_layout()
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/windows_comp_{}_{}_{}_{}.pdf".format(
                attribute, traffic, obs_space, agent
            ),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_obs_comparison(
    trial_number: int,
    agent: str,
    slices_req: dict,
    windows_size_obs: int,
    traffic: str,
    obs_spaces: list,
    runs: int,
    steps_number: int = 2000,
) -> None:
    x_label = "Time (ms)"
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
    obs_names_colors = {
        "full": ("Full", "#003f5c"),
        "partial": ("Partial", "#444e86"),
    }
    data_index = {
        "throughput": (2, "Throughput (Mbps)"),
        "latency": (4, "Latency (ms)"),
        "pkt_loss": (5, "Packet loss rate"),
        "long_term_pkt_thr": (7, "Throughput (Mbps)"),
        "fifth_perc_pkt_thr": (8, "Throughput (Mbps)"),
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(data_index[attribute][1], fontsize=14)
        plt.grid()
        for slice in slices.keys():
            for obs_space in obs_spaces:
                if attribute in slice_requirements[traffic][slice].keys():
                    hist = np.zeros((runs, steps_number))
                    for run_number in range(runs):
                        hist[run_number, :] = Slice.read_hist(
                            "test/{}/ws_{}/{}/{}/run{}".format(
                                agent,
                                windows_size_obs,
                                obs_space,
                                traffic,
                                run_number + 1,
                            ),
                            trial_number,
                            slices[slice],
                        )[data_index[attribute][0]]
                    hist = np.mean(hist, axis=0)
                    hist = (
                        Basestation.packets_to_mbps(8192 * 8, hist)
                        if data_index[attribute][0] in [2, 7, 8]
                        else hist
                    )
                    plt.plot(
                        range(0, len(hist)),
                        hist,
                        label="{}, {} Obs. Space".format(
                            slices_names_markers[slice][0],
                            obs_names_colors[obs_space][0],
                        ),
                        marker=slices_names_markers[slice][1],
                        color=obs_names_colors[obs_space][1],
                        markevery=50,
                    )
            if attribute in slice_requirements[traffic][slice].keys():
                plt.plot(
                    range(steps_number),
                    slices_req[traffic][slice][attribute] * np.ones(steps_number),
                    linestyle="--",
                    marker=slices_names_markers[slice][1],
                    color="blue",
                    markevery=50,
                    zorder=3,
                    label="{} Req.".format(slices_names_markers[slice][0]),
                )
        fig.tight_layout()
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/obs_comp_{}_{}_ws{}_{}.pdf".format(
                attribute, traffic, windows_size_obs, agent
            ),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_reward_ws_comparison(
    trial_number: int,
    agents: list,
    windows_sizes: list,
    traffic: str,
    obs_space: str,
    runs: int,
    steps_number: int = 2000,
) -> None:
    x_label = "Time (ms)"

    agents_names_colors = {
        "a2c": ("A2C", "#003f5c"),
        "ppo": ("PPO", "#444e86"),
        "dqn": ("DQN", "#955196"),
        "rr": ("RR", "#dd5182"),
        "pf": ("PF", "#ff6e54"),
        "mt": ("MT", "#ffa600"),
    }
    ws_markers = {
        1: "*",
        10: "o",
        50: "p",
        100: "d",
    }
    data_index = {
        "reward": 1,
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.grid()
        for windows_size_obs in windows_sizes:
            for agent in agents:
                hist = np.zeros((runs, steps_number))
                for run_number in range(runs):
                    hist[run_number, :] = Basestation.read_hist(
                        "test/{}/ws_{}/{}/{}/run{}".format(
                            agent,
                            windows_size_obs,
                            obs_space,
                            traffic,
                            run_number + 1,
                        ),
                        trial_number,
                    )[data_index[attribute]]
                hist = np.mean(hist, axis=0)
                plt.plot(
                    range(0, len(hist)),
                    hist,
                    label="{}, $W_s$={}".format(
                        agents_names_colors[agent][0], windows_size_obs
                    ),
                    marker=ws_markers[windows_size_obs],
                    color=agents_names_colors[agent][1],
                    markevery=50,
                )
        fig.tight_layout()
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/agents_ws_comp_{}_{}.pdf".format(traffic, obs_space),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


def plot_reward_obs_comparison(
    trial_number: int,
    agents: list,
    windows_size_obs: int,
    traffic: str,
    obs_spaces: list,
    runs: int,
    steps_number: int = 2000,
) -> None:
    x_label = "Time (ms)"

    agents_names_colors = {
        "a2c": ("A2C", "#003f5c"),
        "ppo": ("PPO", "#444e86"),
        "dqn": ("DQN", "#955196"),
        "rr": ("RR", "#dd5182"),
        "pf": ("PF", "#ff6e54"),
        "mt": ("MT", "#ffa600"),
    }
    obs_names_markers = {
        "full": ("Full", "*"),
        "partial": ("Partial", "p"),
    }
    data_index = {
        "reward": 1,
    }

    for attribute in data_index.keys():
        w, h = plt.figaspect(0.6)
        fig = plt.figure(figsize=(w, h))
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.grid()
        for obs_space in obs_spaces:
            for agent in agents:
                hist = np.zeros((runs, steps_number))
                for run_number in range(runs):
                    hist[run_number, :] = Basestation.read_hist(
                        "test/{}/ws_{}/{}/{}/run{}".format(
                            agent,
                            windows_size_obs,
                            obs_space,
                            traffic,
                            run_number + 1,
                        ),
                        trial_number,
                    )[data_index[attribute]]
                hist = np.mean(hist, axis=0)
                plt.plot(
                    range(0, len(hist)),
                    hist,
                    label="{}, {} Obs. Space".format(
                        agents_names_colors[agent][0],
                        obs_names_markers[obs_space][0],
                    ),
                    marker=obs_names_markers[obs_space][1],
                    color=agents_names_colors[agent][1],
                    markevery=50,
                )
        fig.tight_layout()
        plt.legend(fontsize=12)
        os.makedirs("./results", exist_ok=True)
        fig.savefig(
            "./results/agents_obs_comp_{}_ws{}.pdf".format(traffic, windows_size_obs),
            # bbox_inches="tight",
            pad_inches=0,
            format="pdf",
            dpi=1000,
        )
        # plt.show()
        plt.close()


trial_number = 1
agents = ["ppo", "rr", "mt"]
windows_sizes = 1
observations_spaces = "full"
traffics = "light"
runs = 10
slice_requirements = {
    "light": {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    },
    "moderate": {
        "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 2, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
    },
    "heavy": {
        "embb": {"throughput": 30, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 3, "latency": 1, "pkt_loss": 0.001},
        "be": {"long_term_pkt_thr": 15, "fifth_perc_pkt_thr": 5},
    },
}
# plot_agents_comparison(
#     trial_number,
#     agents,
#     slice_requirements,
#     windows_sizes,
#     traffics,
#     observations_spaces,
#     10,
# )

# plot_ws_comparison(
#     trial_number,
#     "ppo",
#     slice_requirements,
#     [1, 50, 100],
#     traffics,
#     observations_spaces,
#     10,
# )

# plot_obs_comparison(
#     trial_number,
#     "ppo",
#     slice_requirements,
#     1,
#     traffics,
#     ["full", "partial"],
#     10,
# )

plot_reward_ws_comparison(1, ["ppo", "mt", "rr"], [10], "light", "full", 1)
# plot_reward_obs_comparison(1, ["ppo"], 50, "light", ["full", "partial"], 10)
