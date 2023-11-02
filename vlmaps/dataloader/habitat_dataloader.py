import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import DictConfig
import hydra

from vlmaps.utils.time_utils import Tic
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf, base_pos2grid_id_3d, grid_id2base_pos_3d, base_rot_mat2theta
from vlmaps.map.map import Map

# from utils.clip_mapping_utils import *
# from utils.clip_utils import *
# from utils.data_collect_utils import *
# from utils.utils import *

from typing import Tuple, List, Dict, Optional, Union


class VLMapsDataloaderHabitat:
    """A pose converter that bridge the pose from habitat simulator to the pose in the map.
    The full map pose is (row, col, angle_deg) where angle_deg is 0 when pointing negative
    row direction (up).
    The cropped map pose is (row, col, angle_deg) whose row and col are rowmin and colmin
    smaller than full map pose's row and col.
    """

    def __init__(
        self,
        data_dir: Union[Path, str],
        map_config: DictConfig,
        map: Map = None,
        load_gt_map: bool = False,
    ):
        """Initialize the pose converter

        Args:
            data_dir (Union[Path, str]): the dir to the data of one scene
            map_config (DictConfig): the config of the map which you can check in config/map_config
            map (Map, optional): The reference to the map if it is already created. When None is provided,
                                    the converter will load the map based on the data_dir and map_config.
                                    Defaults to None.
            load_gt_map (bool, optional): Boolean that controls whether to load GT map. Defaults to False.
        """
        self.data_dir = data_dir
        self.map_config = map_config
        self.map = map
        self.cs = map_config.cell_size
        self.gs = map_config.grid_size
        self.camera_height = map_config.pose_info.camera_height

        if map is None:
            # setup dirs
            self.map_dir = os.path.join(data_dir, map_config.map_type)

            # setup map
            self.map = Map.create(map_config)

            # load map
            load_success = self.map.load_map(data_dir)
            assert (
                load_success == True
            ), f"Map loading fails. It could be because the map hasn't been created at {self.map_dir}."
            self.map.generate_obstacle_map()

        self.obstacles = self.map.obstacles_map
        self.obstacles_cropped = self.map.obstacles_cropped
        self.rmin = self.map.rmin
        self.xmax = self.map.rmax
        self.cmin = self.map.cmin
        self.ymax = self.map.cmax

        self.base2cam_tf, self.base_transform = self.map.base2cam_tf, self.map.base_transform
        self.base_poses = np.loadtxt(self.map.pose_path)
        self.init_base_tf = (
            self.base_transform @ cvt_pose_vec2tf(self.base_poses[0]) @ np.linalg.inv(self.base_transform)
        )
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)

        self.full_map_pose = None  # (row, col, theta_deg)

        # TODO: implement loading GT map option

    def get_obstacles_cropped(self) -> np.array:
        return self.obstacles_cropped

    def get_obstacles_cropped_no_floor(self) -> np.array:
        floor_mask = self.gt_cropped == 2
        obstacles_cropped_no_floor = self.obstacles_cropped.copy()
        obstacles_cropped_no_floor[floor_mask] = 1
        return obstacles_cropped_no_floor

    def get_color_topdown_bgr_cropped(self, times: float = 1) -> np.array:
        color_top_down = self.map.generate_rgb_topdown_map()
        color_top_down = color_top_down[self.rmin : self.xmax + 1, self.cmin : self.ymax + 1]
        color_top_down_bgr = cv2.cvtColor(color_top_down, cv2.COLOR_RGB2BGR)
        color_top_down_bgr = cv2.resize(
            color_top_down_bgr,
            (color_top_down_bgr.shape[1] * times, color_top_down_bgr.shape[1] * times),
        )
        return color_top_down_bgr

    def get_gt_semantic_cropped(self) -> np.array:
        return self.gt_cropped

    def from_cropped_map_pose(self, row: int, col: int, theta_deg: float):
        self.full_map_pose = [row + self.rmin, col + self.cmin, theta_deg]

    def from_full_map_pose(self, row: int, col: int, theta_deg: float):
        self.full_map_pose = [row, col, theta_deg]

    def from_habitat_tf(self, tf_hab: np.ndarray):
        tf = self.inv_init_base_tf @ self.base_transform @ tf_hab @ np.linalg.inv(self.base_transform)
        theta = base_rot_mat2theta(tf[:3, :3])
        theta_deg = np.rad2deg(theta)
        x, y, z = tf[:3, 3]
        row, col, height = base_pos2grid_id_3d(self.gs, self.cs, x, y, z)
        self.full_map_pose = [row, col, theta_deg]

    def from_camera_tf(self, tf_cam: np.ndarray):
        tf_hab = self.base_transform @ self.inv_init_base_tf @ self.base2cam_tf @ tf_cam
        self.from_habitat_tf(tf_hab)

    def to_cropped_map_pose(self) -> Tuple[int, int, float]:
        assert self.full_map_pose is not None, "Please call from_xx() first."
        return [self.full_map_pose[0] - self.rmin, self.full_map_pose[1] - self.cmin, self.full_map_pose[2]]

    def to_full_map_pose(self) -> Tuple[int, int, float]:
        assert self.full_map_pose is not None, "Please call from_xx() first."
        return self.full_map_pose

    def to_habitat_tf(self) -> np.ndarray:
        assert self.full_map_pose is not None, "Please call from_xx() first."
        row, col, theta_deg = self.full_map_pose
        x, y, z = grid_id2base_pos_3d(row, col, 0, self.cs, self.gs)
        theta = np.deg2rad(theta_deg)
        tf = np.eye(4)
        tf[:3, 3] = [x, y, z]
        tf[0, 0] = np.cos(theta)
        tf[1, 1] = np.cos(theta)
        tf[0, 1] = -np.sin(theta)
        tf[1, 0] = np.sin(theta)
        tf_hab = np.linalg.inv(self.base_transform) @ self.init_base_tf @ tf @ self.base_transform
        return tf_hab


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    dataloader = VLMapsDataloaderHabitat(data_dirs[0], config.map_config)
    ids_list = np.random.randint(0, len(dataloader.base_poses), 10)
    for test_i, i in enumerate(ids_list):
        base_hab_tf = cvt_pose_vec2tf(dataloader.base_poses[i])
        dataloader.from_habitat_tf(base_hab_tf)
        full_map_pose = dataloader.to_full_map_pose()
        print("full map pose: ", full_map_pose)
        dataloader.from_full_map_pose(*full_map_pose)
        cvt_hab_tf = dataloader.to_habitat_tf()
        print("The correct habitat tf is: \n", base_hab_tf)
        print("The converted habitat tf is: \n", cvt_hab_tf)

        err = np.linalg.norm(base_hab_tf - cvt_hab_tf)
        print(f"Error: {err}")
        assert err < 1, "[TEST RESULTS]: FAIL! The converted habitat tf is not correct."
        print(f"[TEST RESULTS]: PASS {test_i+1}/{len(ids_list)}!")


if __name__ == "__main__":
    main()
