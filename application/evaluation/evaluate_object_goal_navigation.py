import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra

from vlmaps.task.habitat_object_nav_task import HabitatObjectNavigationTask
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.matterport3d_categories import mp3dcat


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = HabitatLanguageRobot(config)
    object_nav_task = HabitatObjectNavigationTask(config)
    object_nav_task.reset_metrics()
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())
        object_nav_task.setup_scene(robot.vlmaps_dataloader)
        object_nav_task.load_task()

        for task_id in range(len(object_nav_task.task_dict)):
            object_nav_task.setup_task(task_id)
            object_categories = parse_object_goal_instruction(object_nav_task.instruction)
            print(f"instruction: {object_nav_task.instruction}")
            robot.empty_recorded_actions()
            robot.set_agent_state(object_nav_task.init_hab_tf)

            for cat_i, cat in enumerate(object_categories):
                print(f"Navigating to category {cat}")
                actions_list = robot.move_to_object(cat)

            recorded_actions_list = robot.get_recorded_actions()
            robot.set_agent_state(object_nav_task.init_hab_tf)
            for action in recorded_actions_list:
                object_nav_task.test_step(robot.sim, action, vis=config.nav.vis)

            save_dir = robot.vlmaps_dataloader.data_dir / (config.map_config.map_type + "_obj_nav_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / f"{task_id:02}.json"
            object_nav_task.save_single_task_metric(save_path)


if __name__ == "__main__":
    main()
