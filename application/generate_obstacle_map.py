import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
import open3d as o3d
import cv2
from scipy.ndimage import distance_transform_edt
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.robot.lang_robot import LangRobot


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    robot = LangRobot(config.params)
    # generate obstacles map based on occupancy within a height range
    robot.load_scene_map(data_dirs[config.scene_id], config.map_config)
    obs_map = robot.map.obstacles_cropped
    obs_map = obs_map.astype(np.uint8) * 255
    cv2.imshow("obs_map", obs_map)
    cv2.waitKey()

    # customize obstacles map
    robot.map.customize_obstacle_map(
        config.map_config.potential_obstacle_names, config.map_config.obstacle_names, vis=True
    )


if __name__ == "__main__":
    main()
