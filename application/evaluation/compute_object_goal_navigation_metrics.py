import os
from pathlib import Path
import numpy as np
import json
from omegaconf import DictConfig
import hydra
from typing import List, Dict, Union


def load_metric(filepath: Union[str, Path]):
    with open(filepath, "r") as f:
        data = json.load(f)

        num_subgoals = data["num_subgoals"]
        num_success_subgoals = len(data["finished_subgoal_ids"])
        success = num_success_subgoals == 4

    return num_subgoals, success, num_success_subgoals, data


def compute_metric(data_dir: Union[str, Path], scene_ids: List[int], metric_folder_name="vlmap_obj_nav_results"):
    vlmaps_data_save_dirs = [
        data_dir / x for x in sorted(os.listdir(data_dir)) if x != ".DS_Store"
    ]  # ignore artifact generated in MacOS
    vlmaps_data_save_dirs = [vlmaps_data_save_dirs[i] for i in scene_ids]
    n_tot_tasks = 0
    n_tot_subgoals = 0
    n_suc_tasks = 0
    n_suc_subgoals = 0
    per_class_suc_dict = {}
    subgoals_suc_dict = {}

    for scene_i, vlmaps_data_dir in enumerate(vlmaps_data_save_dirs):
        metric_save_dir = vlmaps_data_dir / metric_folder_name

        metric_paths_list = sorted(os.listdir(metric_save_dir))

        metric_paths_list = [metric_save_dir / x for x in metric_paths_list if x != ".DS_Store"]

        for metric_i, metric_path in enumerate(metric_paths_list):
            print("metric_path: ", metric_path)
            n_sub, suc, n_suc_sub, data = load_metric(metric_path)
            # nav_info = NavInfo()
            # nav_info.load_metric_details(metric_path)
            for subgoal_i, subgoal in enumerate(data["goal_classes"]):
                if subgoal not in per_class_suc_dict:
                    per_class_suc_dict[subgoal] = {"finished": 0, "all": 0}
                per_class_suc_dict[subgoal]["all"] += 1
                if subgoal_i in data["finished_subgoal_ids"]:
                    per_class_suc_dict[subgoal]["finished"] += 1

            n_finished_continue_subgoals = 0
            for i in range(4):
                if i in data["finished_subgoal_ids"]:
                    n_finished_continue_subgoals += 1
                    continue
                break

            for i in range(n_finished_continue_subgoals, -1, -1):
                if i not in subgoals_suc_dict:
                    subgoals_suc_dict[i] = 0
                subgoals_suc_dict[i] += 1

            if n_suc_sub != len(data["finished_subgoal_ids"]):
                print(metric_path)
                print(n_suc_sub, data["finished_subgoal_ids"])

            if n_sub != 4:
                print(n_sub, suc, n_suc_sub)
                print(metric_path)
            n_tot_tasks += 1
            n_tot_subgoals += n_sub
            n_suc_tasks += suc
            # n_suc_subgoals += n_suc_sub
            n_suc_subgoals += len(data["finished_subgoal_ids"])

    sr = float(n_suc_tasks) / n_tot_tasks
    sub_sr = float(n_suc_subgoals) / n_tot_subgoals
    print(f"Total tasks number is {n_tot_tasks}.")
    print(f"Total subgoals number is {n_tot_subgoals}.")
    print(f"Task success rate is {sr}")
    print(f"Subgoal success rate is {sub_sr}")
    for k, v in per_class_suc_dict.items():
        finished = v["finished"]
        all = v["all"]
        print(f"{k} {finished}/{all}")
    # for n_subgoal in subgoals_suc_dict.keys():
    for n_subgoal in range(5):
        if n_subgoal not in subgoals_suc_dict:
            break
        print(
            f"The success tasks number for finishing {n_subgoal} subgoals in one row: {subgoals_suc_dict[n_subgoal]}"
            f"({float(subgoals_suc_dict[n_subgoal])/n_tot_tasks})"
        )
    return sr, sub_sr


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    compute_metric(data_dir, scene_ids)


if __name__ == "__main__":
    main()
