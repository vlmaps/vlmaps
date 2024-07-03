from pathlib import Path
import cv2
import hydra
from omegaconf import DictConfig
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.map.interactive_map import InteractiveMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.mapping_utils import (
    cvt_pose_vec2tf,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    robot.map.init_categories(mp3dcat[1:-1])
