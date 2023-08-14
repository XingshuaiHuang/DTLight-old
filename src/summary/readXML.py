import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_utils.utils import insert_data_to_json


# Solve the problem of "ModuleNotFoundError: No module named 'xxx'"
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.map_config import map_configs

"""
Compute average delay, average waiting time, and average duration.
"""


def readXML(
    log="logs",
    plot=True,
    print_delay=False,
    save_result=True,
    task=None,
    print_task=False,
    print_history=False,
):
    root_dir = "."
    log_dir = root_dir + os.sep + log + os.sep
    env_base = root_dir + os.sep + "environments" + os.sep

    if not task:
        names = [
            folder for folder in next(os.walk(log_dir))[1]
        ]  # Go through all the saved results in log_dir
        metrics = ["timeLoss", "duration", "waitingTime"]  # timeLoss (var in tripinfo) = delay
    else:
        names = task  # Go through selected results
        metrics = ["timeLoss"]

    epochs = 100
    current_delay = None

    for metric in metrics:
        output_file = "src/summary/avg_{}.json".format(
            metric
        )
        run_avg = dict()

        for name in names:
            split_name = name.split("-")
            map_name = split_name[2]

            # ------------------- Calculate average metric per episode ------------------
            average_per_episode = []
            for i in range(1, epochs):
                trip_file_name = log_dir + name + os.sep + "tripinfo_" + str(i) + ".xml"
                if not os.path.exists(trip_file_name):
                    if not task:
                        print("No " + trip_file_name)
                    break
                try:
                    tree = ET.parse(trip_file_name)
                    root = tree.getroot()
                    num_trips, total = 0, 0.0
                    last_departure_time = 0
                    last_depart_id = ""
                    for child in root:
                        try:
                            num_trips += 1
                            total += float(child.attrib[metric])
                            if metric == "timeLoss":
                                total += float(child.attrib["departDelay"])
                                depart_time = float(child.attrib["depart"])
                                if depart_time > last_departure_time:
                                    last_departure_time = depart_time
                                    last_depart_id = child.attrib["id"]
                        except Exception as e:
                            # raise e
                            break

                    route_file_name = (
                        env_base
                        + map_name
                        + os.sep
                        + map_name
                        + "_"
                        + str(i)
                        + ".rou.xml"
                    )

                    if metric == "timeLoss":  # Calc. departure delays
                        try:
                            tree = ET.parse(route_file_name)
                        except FileNotFoundError:
                            route_file_name = (
                                env_base + map_name + os.sep + map_name + ".rou.xml"
                            )
                            tree = ET.parse(route_file_name)
                        root = tree.getroot()
                        last_departure_time = None
                        for child in root:
                            if child.attrib["id"] == last_depart_id:
                                last_departure_time = float(
                                    child.attrib["depart"]
                                )  # Get the time it was suppose to depart
                        never_departed = []
                        if last_departure_time is None:
                            raise Exception("Wrong trip file")
                        for child in root:
                            if child.tag != "vehicle":
                                continue
                            depart_time = float(child.attrib["depart"])
                            if depart_time > last_departure_time:
                                never_departed.append(depart_time)
                        never_departed = np.asarray(never_departed)
                        never_departed_delay = np.sum(
                            float(map_configs[map_name]["end_time"]) - never_departed
                        )
                        total += never_departed_delay
                        num_trips += len(never_departed)

                    average = total / num_trips
                    average_per_episode.append(average)
                except ET.ParseError as e:
                    raise e
                    break
            # -----------------------------------------------------------------------------

            # method_name = split_name[0].split('_')[0] + '_' + split_name[0].split('_')[1]
            run_name = (split_name[0]
                + " "
                + split_name[2]  # map name
                + " "
                + split_name[3]  # len(lights)
                + " "
                + split_name[4]  # state
                + " "
                + split_name[5]  # reward
            )
            average_per_episode = np.asarray(average_per_episode)

            if run_name in run_avg:
                run_avg[run_name].append(average_per_episode)
            else:
                run_avg[run_name] = [average_per_episode]

        # ------------------- Calculate mean and std ------------------
        alg_res = []
        alg_name = []
        for run_name in run_avg:
            list_runs = run_avg[run_name]
            min_len = min([len(run) for run in list_runs])
            list_runs = [run[:min_len] for run in list_runs]
            avg_delays = np.sum(list_runs, 0) / len(list_runs)
            err = np.std(list_runs, axis=0)

            alg_name.append(run_name)
            alg_res.append(avg_delays)

            alg_name.append(run_name + "_yerr")
            alg_res.append(err)

            if metric == "timeLoss":
                current_delay = avg_delays[-1]
                current_eps = len(avg_delays)
                min_delay = np.min(avg_delays)
                min_eps = np.argmin(avg_delays) + 1

                if print_task:
                    print("\n", run_name)
                if print_delay:
                    print(" ", "-" * 50)
                    if print_history:
                        print(f"    Avg. delay per episode: {avg_delays}")
                    else:
                        print(
                            "    Current episode {} | avg_delay: {:.3f}".format(
                                current_eps, current_delay
                            )
                        )
                    print(
                        "    Min avg_delay episode: {} | avg_delay: {:.3f}".format(
                            min_eps, min_delay)
                        )
                    print(" ", "-" * 50, "\n")

            if plot:
                if not task:
                    plt.title(run_name)
                    plt.plot(avg_delays)
                    plt.show()
                else:
                    plt.plot(avg_delays, label=run_name.split()[0])
                    plt.legend(fontsize=8)
        plt.show()

        if save_result:
            np.set_printoptions(threshold=sys.maxsize)
            new_rest = dict()
            for i, res in enumerate(alg_res):
                new_rest[alg_name[i]] = res.tolist()
            insert_data_to_json(output_file, new_rest, long_list=True)

    return min_delay, min_eps, current_delay, current_eps


if __name__ == "__main__":
    readXML()
