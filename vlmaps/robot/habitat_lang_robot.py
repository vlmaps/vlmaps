import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from omegaconf import DictConfig
import hydra

# function to display the topdown map
import habitat_sim
import cv2
import open3d as o3d

from vlmaps.robot.lang_robot import LangRobot
from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from vlmaps.navigator.navigator import Navigator
from vlmaps.controller.discrete_nav_controller import DiscreteNavController

from vlmaps.utils.mapping_utils import (
    grid_id2base_pos_3d,
    grid_id2base_pos_3d_batch,
    base_pos2grid_id_3d,
    cvt_pose_vec2tf,
)
from vlmaps.utils.index_utils import find_similar_category_id
from vlmaps.utils.habitat_utils import make_cfg, tf2agent_state, agent_state2tf, display_sample
from vlmaps.utils.matterport3d_categories import mp3dcat

from typing import List, Tuple, Dict, Any, Union


class HabitatLanguageRobot(LangRobot):
    """
    This class inherits the LangRobot interface for the user.
    It also handles the interface with the Habitat simulator,
    the data, and the control of agent in the environment.
    """

    # the next line should go in the application function that creates this class
    # @hydra.main(version_base=None, config_path="../config/habitat", config_name="config")
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.test_scene_dir = self.config["data_paths"]["habitat_scene_dir"]
        data_dir = Path(self.config["data_paths"]["vlmaps_data_dir"]) / "vlmaps_dataset"
        self.vlmaps_data_save_dirs = [
            data_dir / x for x in sorted(os.listdir(data_dir)) if x != ".DS_Store"
        ]  # ignore artifact generated in MacOS
        self.map_type = self.config["params"]["map_type"]
        self.camera_height = self.config["params"]["camera_height"]
        self.gs = self.config["params"]["gs"]
        self.cs = self.config["params"]["cs"]
        self.forward_dist = self.config["params"]["forward_dist"]
        self.turn_angle = self.config["params"]["turn_angle"]

        self.sim: habitat_sim.Simulator
        self.sim = None
        self.last_scene_name = ""
        # self.agent_model_dir = self.config["data_paths"]["agent_mesh_dir"]

        self.vis = False

        self.nav = Navigator()
        self.controller = DiscreteNavController(self.config["params"]["controller_config"])

    def setup_scene(self, scene_id: int):
        """
        Setup the simulator, load scene data and prepare
        the LangRobot interface for navigation
        """
        self.scene_id = scene_id
        vlmaps_data_dir = self.vlmaps_data_save_dirs[scene_id]
        print(vlmaps_data_dir)
        self.scene_name = vlmaps_data_dir.name.split("_")[0]

        self._setup_sim(self.scene_name)

        self.setup_map(vlmaps_data_dir)

        cropped_obst_map = self.map.get_obstacle_cropped()
        if self.config.map_config.potential_obstacle_names and self.config.map_config.obstacle_names:
            print("come here")
            self.map.customize_obstacle_map(
                self.config.map_config.potential_obstacle_names,
                self.config.map_config.obstacle_names,
                vis=self.config.nav.vis,
            )
            cropped_obst_map = self.map.get_customized_obstacle_cropped()

        self.nav.build_visgraph(
            cropped_obst_map,
            self.vlmaps_dataloader.rmin,
            self.vlmaps_dataloader.cmin,
            vis=self.config["nav"]["vis"],
        )

        # self._setup_localizer(vlmaps_data_dir)

    def setup_map(self, vlmaps_data_dir: str):
        self.load_scene_map(vlmaps_data_dir, self.config["map_config"])

        # TODO: check if needed
        if "3d" in self.config.map_config.map_type:
            self.map.init_categories(mp3dcat.copy())
            self.global_pc = grid_id2base_pos_3d_batch(self.map.grid_pos, self.cs, self.gs)

        self.vlmaps_dataloader = VLMapsDataloaderHabitat(vlmaps_data_dir, self.config.map_config, map=self.map)

    def _setup_sim(self, scene_name: str):
        """
        Setup Habitat simulator, load habitat scene and relevant mesh data
        """
        if self.sim is not None:
            self.sim.close()
        self.test_scene = os.path.join(self.test_scene_dir, scene_name, scene_name + ".glb")
        self.sim_setting = {
            "scene": self.test_scene,
            **self.config["params"]["sim_setting"],
        }

        cfg = make_cfg(self.sim_setting)

        # create a simulator instance
        if self.sim is None or scene_name == self.last_scene_name:
            self.sim = habitat_sim.Simulator(cfg)
            agent = self.sim.initialize_agent(self.sim_setting["default_agent"])
        else:
            self.sim.reconfigure(cfg)
        self.last_scene_name = scene_name

        # TODO: add in document to enable this features
        # load agent mesh for visualization
        # if self.agent_model_dir:
        #     rigid_obj_mgr = self.sim.get_rigid_object_manager()
        #     obj_attr_mgr = self.sim.get_object_template_manager()
        #     locobot_template_id = obj_attr_mgr.load_configs(self.agent_model_dir)[0]
        #     locobot = rigid_obj_mgr.add_object_by_template_id(locobot_template_id, self.sim.agents[0].scene_node)

    def set_agent_state(self, tf: np.ndarray):
        agent_state = tf2agent_state(tf)
        self.sim.get_agent(0).set_state(agent_state)
        self._set_nav_curr_pose()

    def get_agent_tf(self) -> np.ndarray:
        agent_state = self.sim.get_agent(0).get_state()
        return agent_state2tf(agent_state)

    def load_gt_region_map(self, region_gt: List[Dict[str, np.ndarray]]):
        obst_cropped = self.vlmaps_dataloader.get_obstacles_cropped()
        self.region_categories = sorted(list(region_gt.keys()))
        self.gt_region_map = np.zeros(
            (len(self.region_categories), obst_cropped.shape[0], obst_cropped.shape[1]), dtype=np.uint8
        )

        for cat_i, cat in enumerate(self.region_categories):
            for box_i, box in enumerate(region_gt[cat]):
                center = np.array(box["region_center"])
                size = np.array(box["region_size"])
                top_left = center - size / 2
                bottom_right = center + size / 2
                top_left_2d = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list([top_left])[0]
                bottom_right_2d = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list(
                    [bottom_right]
                )[0]

                self.gt_region_map[cat_i] = cv2.rectangle(
                    self.gt_region_map[cat_i],
                    (int(top_left_2d[1]), int(top_left_2d[0])),
                    (int(bottom_right_2d[1]), int(bottom_right_2d[0])),
                    1,
                    -1,
                )

    def get_distribution_map(
        self, name: str, scores: np.ndarray, pos_list_cropped: List[List[float]], decay_rate: float = 0.1
    ):
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        for pos_i, pos in enumerate(pos_list_cropped):
            # dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            pos = np.round(pos[0]), np.round(pos[1])
            tmp_dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]

            con = scores[pos_i]
            dists = distance_transform_edt(tmp_dist_map == 0)
            reduct = con * dists * decay_rate
            tmp = np.ones_like(tmp_dist_map) * con - reduct
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map += tmp_dist_map
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_distribution_map_3d(
        self, name: str, scores: np.ndarray, pos_list_3d: List[List[float]], decay_rate: float = 0.1
    ):
        """
        pos_list_3d: list of 3d positions in 3d map coordinate
        """
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        sim_mat = np.zeros((self.global_pc.shape[0], len(scores)))
        for pro_i, (con, pos) in enumerate(zip(scores, pos_list_3d)):
            print("confidence: ", con)

            dists = np.linalg.norm(self.global_pc[:, [0, 2]] - pos[[0, 2]], axis=1) / self.vlmaps_dataloader.cs
            sim = np.clip(con - decay_rate * dists, 0, 1)
            sim_mat[:, pro_i] = sim

        sim = np.max(sim_mat, axis=1)
        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(sim, name=name)

        return sim.flatten()

    def get_vl_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict_mask = self.map.get_predict_mask(name)
        predict_mask = predict_mask.astype(np.float32)
        predict_mask = (gaussian_filter(predict_mask, sigma=1) > 0.5).astype(np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(predict_mask, name=name + "_predict_mask")
            self._vis_dist_map(dist_map, name=name + f"_{decay_rate}")
        return dist_map

    def get_vl_distribution_map_3d(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict = np.argmax(self.map.scores_mat, axis=1)
        i = find_similar_category_id(name, self.map.categories)
        sim = predict == i

        target_pc = self.global_pc[sim, :]
        other_ids = np.where(sim == 0)[0]
        other_pc = self.global_pc[other_ids, :]
        target_sim = np.ones((target_pc.shape[0], 1))
        other_sim = np.zeros((other_pc.shape[0], 1))
        for other_p_i, p in enumerate(other_pc):
            dist = np.linalg.norm(target_pc - p, axis=1) / self.cs
            min_dist_i = np.argmin(dist)
            min_dist = dist[min_dist_i]
            other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

        new_pc_global = self.global_pc.copy()
        new_sim = np.ones((new_pc_global.shape[0], 1), dtype=np.float32)
        for s_i, s in enumerate(other_sim):
            new_sim[other_ids[s_i]] = s

        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(new_sim, name=name)
        return new_sim.flatten()

    def get_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        if self.area_map_type == "clip_sparse":
            return self.get_clip_sparse_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "concept_fusion":
            return self.get_concept_fusion_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "lseg":
            return self.get_lseg_region_map(name, decay_rate)
        elif self.area_map_type == "gt":
            return self.get_gt_region_map(name, decay_rate)

    def get_gt_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map_type == "gt"
        id = find_similar_category_id(name, self.region_categories)
        predict_mask = self.gt_region_map[id]

        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_lseg_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.map_type == "lseg"
        assert self.area_map_type == "lseg"
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.map.get_region_predict_mask(name)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_concept_fusion_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.area_map.get_predict_mask(name)
        # print("predict_mask: ", predict_mask.shape)
        # mask_vis = cv2.cvtColor((predict_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # cv2.imshow("mask_vis", mask_vis)
        # cv2.waitKey()
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_clip_sparse_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        scores = self.area_map.get_scores(name)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        robot_pose_list = self.area_map.get_robot_pose_list()
        ids = np.argsort(-scores.flatten())
        # max_id = np.argmax(scores)
        # obst_map = self.vlmaps_dataloader.get_obstacles_cropped_no_floor()
        # obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)

        for i, tf_hab in enumerate(robot_pose_list):
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            row, col, deg = self.vlmaps_dataloader.convert_habitat_tf_to_cropped_map_pose(tf_hab)
            if row < 0 or row >= dist_map.shape[0] or col < 0 or col >= dist_map.shape[1]:
                continue
            # print(i, tf_hab[:3, 3].flatten(), row, col)
            # obst_map[int(row), int(col)] = (0, 255, 0)
            # cv2.circle(obst_map, (int(col), int(row)), 3, (0, 255, 0), -1)
            # cv2.imshow("obst_map", obst_map)
            # cv2.waitKey(1)
            s = scores[i]
            tmp_dist_map[row, col] = s
            dists = distance_transform_edt(tmp_dist_map == 0)
            tmp = np.ones_like(dists) * s - (dists * decay_rate)
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map = np.where(dist_map > tmp_dist_map, dist_map, tmp_dist_map)

        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.01)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.01)

    def get_major_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.1)

    def get_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.03)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.05)

    def get_major_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None and intr_mat is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.01)

    def _vis_dist_map(self, dist_map: np.ndarray, name: str = ""):
        obst_map = self.vlmaps_dataloader.get_obstacles_cropped_no_floor()
        obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)
        target_heatmap = show_cam_on_image(obst_map.astype(float) / 255.0, dist_map)
        cv2.imshow(f"heatmap_{name}", target_heatmap)

    def _vis_dist_map_3d(self, heatmap: np.ndarray, transparency: float = 0.3, name: str = ""):
        print(f"heatmap of {name}")
        sim_new = (heatmap * 255).astype(np.uint8)
        rgb_pc = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
        rgb_pc = rgb_pc.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
        rgb_pc = rgb_pc * transparency + self.map.grid_rgb / 255.0 * (1 - transparency)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.global_pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
        o3d.visualization.draw_geometries([pcd])

    def get_max_pos(self, map: np.ndarray) -> Tuple[float, float]:
        id = np.argmax(map)
        row, col = np.unravel_index(id, map.shape)

        if self.config["nav"]["vis"]:
            self._vis_dist_map(map, name="fuse")
        return row + self.vlmaps_dataloader.rmin, col + self.vlmaps_dataloader.cmin

    def get_max_pos_3d(self, heat: np.ndarray) -> Tuple[float, float, float]:
        id = np.argmax(heat)
        grid_map_pos_3d = self.map.grid_pos[id]
        return grid_map_pos_3d

    def move_to(self, pos: Tuple[float, float]) -> List[str]:
        """Move the robot to the position on the full map
            based on accurate localization in the environment

        Args:
            pos (Tuple[float, float]): (row, col) on full map

        Returns:
            List[str]: list of actions
        """
        actual_actions_list = []
        success = False
        # while not success:
        self._set_nav_curr_pose()
        curr_pose_on_full_map = self.get_agent_pose_on_map()  # (row, col, angle_deg) on full map
        paths = self.nav.plan_to(
            curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["vis"]
        )  # take (row, col) in full map
        print(paths)
        actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, paths[1:])
        success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["vis"])
        actual_actions_list.extend(real_actions_list)

        actual_actions_list.append("stop")

        if not hasattr(self, "recorded_actions_list"):
            self.recorded_actions_list = []
        self.recorded_actions_list.extend(actual_actions_list)

        return actual_actions_list

    def turn(self, angle_deg: float):
        """
        Turn right a relative angle in degrees
        """
        if angle_deg < 0:
            actions_list = ["turn_left"] * int(np.abs(angle_deg / self.turn_angle))
        else:
            actions_list = ["turn_right"] * int(angle_deg / self.turn_angle)

        success, real_actions_list = self.execute_actions(actions_list, vis=self.config.nav.vis)

        self.recorded_actions_list.extend(real_actions_list)
        return real_actions_list

    def execute_actions(
        self,
        actions_list: List[str],
        poses_list: List[List[float]] = None,
        vis: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Execute actions and check
        """
        if poses_list is not None:
            assert len(actions_list) == len(poses_list)
        if vis:
            map = self.map.get_customized_obstacle_cropped()
            map = (map[:, :, None] * 255).astype(np.uint8)
            map = np.tile(map, [1, 1, 3])
            self.display_goals_on_map(map, 3, (0, 255, 0))
            if hasattr(self, "recorded_robot_pos") and len(self.recorded_robot_pos) > 0:
                map = self.display_full_map_pos_list_on_map(map, self.recorded_robot_pos)
            else:
                self.recorded_robot_pos = []

        real_actions_list = []
        for action_i, action in enumerate(actions_list):
            self._execute_action(action)

            real_actions_list.append(action)
            if vis:
                self.display_obs(waitkey=True)
                self.display_curr_pos_on_map(map)
            if poses_list is None:
                continue
            row, col, angle = self._get_full_map_pose()
            if vis:
                self.recorded_robot_pos.append((row, col))
            # x, z = grid_id2base_pos_3d(self.gs, self.cs, col, row)
            # pred_x, pred_z, pred_angle = poses_list[action_i]
            # success = self._check_if_pose_match_prediction(x, z, pred_x, pred_z)
            # if not success:
            #     return success, real_actions_list
        return True, real_actions_list

    def pass_goal_bboxes(self, goal_bboxes: Dict[str, Any]):
        self.goal_bboxes = goal_bboxes

    def pass_goal_tf(self, goal_tfs: List[np.ndarray]):
        self.goal_tfs = goal_tfs

    def pass_goal_tf_list(self, goal_tfs: List[List[np.ndarray]]):
        self.all_goal_tfs = goal_tfs
        self.goal_id = 0

    def _execute_action(self, action: str):
        self.sim.step(action)

    def _check_if_pose_match_prediction(self, real_x: float, real_z: float, pred_x: float, pred_z: float):
        dist_thres = self.forward_dist
        dx = pred_x - real_x
        dz = pred_z - real_z
        dist = np.sqrt(dx * dx + dz * dz)
        return dist < dist_thres

    def _set_nav_curr_pose(self):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        """
        agent_state = self.sim.get_agent(0).get_state()
        hab_tf = agent_state2tf(agent_state)
        self.vlmaps_dataloader.from_habitat_tf(hab_tf)
        row, col, angle_deg = self.vlmaps_dataloader.to_full_map_pose()
        self.curr_pos_on_map = (row, col)
        self.curr_ang_deg_on_map = angle_deg
        print("set curr pose: ", row, col, angle_deg)

    def _get_full_map_pose(self) -> Tuple[float, float, float]:
        agent_state = self.sim.get_agent(0).get_state()
        hab_tf = agent_state2tf(agent_state)
        self.vlmaps_dataloader.from_habitat_tf(hab_tf)
        row, col, angle_deg = self.vlmaps_dataloader.to_full_map_pose()
        return row, col, angle_deg

    def display_obs(self, waitkey: bool = False):
        obs = self.sim.get_sensor_observations(0)
        display_sample(self.sim_setting, obs["color_sensor"], waitkey=waitkey)

    def display_curr_pos_on_map(self, map: np.ndarray):
        row, col, angle = self._get_full_map_pose()
        self.vlmaps_dataloader.from_full_map_pose(row, col, angle)
        row, col, angle = self.vlmaps_dataloader.to_cropped_map_pose()
        map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        cv2.imshow("real path", map)
        cv2.waitKey(1)

    def display_full_map_pos_list_on_map(self, map: np.ndarray, pos_list: List[List[float]]) -> np.ndarray:
        for pos_i, pos in enumerate(pos_list):
            self.vlmaps_dataloader.from_full_map_pose(pos[0], pos[1], 0)
            row, col, _ = self.vlmaps_dataloader.to_cropped_map_pose()
            map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        return map

    def display_goals_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if not hasattr(self, "goal_tfs") and not hasattr(self, "goal_bboxes"):
            return map
        if hasattr(self, "goal_tfs"):
            map = self.display_goal_tfs_on_map(map, radius_pix, color)
        elif hasattr(self, "goal_bboxes"):
            map = self.display_goal_bboxes_on_map(map, color)
        else:
            print("no goal tfs or bboxes passed to the robot.")
        return map

    def display_goal_bboxes_on_map(
        self,
        map: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        centers = self.goal_bboxes["centers"]
        sizes = self.goal_bboxes["sizes"]
        cs = self.vlmaps_dataloader.cs
        centers_cropped = self.vlmaps_dataloader.convert_habitat_pos_list_to_cropped_map_pos_list(centers)
        for center_i, center in enumerate(centers_cropped):
            size = [float(x) / cs for x in sizes[center_i]]  # in habitat robot coord
            size = size[[2, 0]]
            min_corner, max_corner = get_bbox(center, size)
            rmin, rmax = int(min_corner[0]), int(max_corner[0])
            cmin, cmax = int(min_corner[1]), int(max_corner[1])
            cv2.rectangle(map, (cmin, rmin), (cmax, rmax), color, 2)

        return map

    def display_goal_tfs_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if self.goal_tfs is None:
            if self.all_goal_tfs is None:
                print("no goal tfs passed to the robot")
                return map
            self.goal_tfs = self.all_goal_tfs[self.goal_id]
            self.goal_id += 1

        for tf_i, tf in enumerate(self.goal_tfs):
            self.vlmaps_dataloader.from_habitat_tf(tf)
            (row, col, angle_deg) = self.vlmaps_dataloader.to_cropped_map_pose()
            cv2.circle(map, (col, row), radius_pix, color, 2)
        self.goal_tfs = None
        return map


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[0])
    robot.set_agent_state(hab_tf)
    obs = robot.sim.get_sensor_observations(0)
    rgb = obs["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("scr rgb", bgr)
    cv2.waitKey()

    tar_hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[800])
    robot.set_agent_state(tar_hab_tf)
    obs = robot.sim.get_sensor_observations(0)
    rgb = obs["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("tar rgb", bgr)
    cv2.waitKey()

    robot.set_agent_state(hab_tf)
    robot.vlmaps_dataloader.from_habitat_tf(tar_hab_tf)
    tar_row, tar_col, tar_angle_deg = robot.vlmaps_dataloader.to_full_map_pose()
    robot.empty_recorded_actions()
    robot.pass_goal_tf([tar_hab_tf])
    robot.move_to((tar_row, tar_col))


if __name__ == "__main__":
    main()
