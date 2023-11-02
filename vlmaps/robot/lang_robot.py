import os
import pickle
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

from vlmaps.map.map import Map

from typing import List, Tuple, Dict, Any


class LangRobot:
    """
    This class provides all primitives API that the robot can call during navigation
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.curr_pos_on_map = None
        self.curr_ang_deg_on_map = None
        pass

    def get_vlmaps_data_dirs(self):
        return self.vlmaps_data_save_dirs

    def load_scene_map(self, data_dir: str, map_config: DictConfig):
        self.map = Map.create(map_config)
        self.map.load_map(data_dir)
        self.map.generate_obstacle_map()

    def empty_recorded_actions(self):
        self.recorded_actions_list = []
        self.recorded_robot_pos = []
        self.goal_tfs = None
        self.all_goal_tfs = None
        self.goal_id = None

    def get_recorded_actions(self):
        return self.recorded_actions_list

    def load_code(self, code_dir: str, task_i: int):
        code_path = os.path.join(code_dir, f"{task_i:06}.txt")
        with open(code_path, "r") as f:
            code = f.readlines()
        code = "".join(code)
        return code

    def _set_nav_curr_pose(self):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        """
        return NotImplementedError

    def execute_actions(self, actions_list: List[Any]):
        return NotImplementedError

    def _execute_action(self, action: str):
        return NotImplementedError

    def get_sound_pos(self, name: str):
        return NotImplementedError
        # tfs = self.sound_map.get_pos(name)

    def move_to(self, pos: Tuple[float, float]):
        """
        Move the robot to the position on the map
        based on accurate localization in the environment
        """
        # check if the pos is None
        return NotImplementedError

    def turn(self, angle_deg: float):
        return NotImplementedError
        # actions_list = self.nav.turn(angle_deg)
        # self.execute_actions(actions_list)
        # self.actions_list += actions_list

    def get_agent_pose_on_map(self) -> Tuple[float, float, float]:
        """
        Return row, col, angle_degree on the full map
        """
        return (
            self.curr_pos_on_map[0],
            self.curr_pos_on_map[1],
            self.curr_ang_deg_on_map,
        )

    def get_pos(self, name: str):
        """
        Return nearest object position on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            return self.curr_pos_on_map
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        center = centers[nearest_id]
        return center

    def get_contour(self, name: str) -> List[List[float]]:
        """
        Return nearest object contour points on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            assert False
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]
            contours = [contours[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        contour = contours[nearest_id]
        return contour

    def with_object_on_left(self, name: str):
        self.face(name)
        self.turn(90)

    def with_object_on_right(self, name: str):
        self.face(name)
        self.turn(-90)

    def move_to_left(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_left_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_right(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_right_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_in_between(self, name_a: str, name_b: str):
        self._set_nav_curr_pose()
        pos = self.map.get_pos_in_between(self.curr_pos_on_map, self.curr_ang_deg_on_map, name_a, name_b)
        self.move_to(pos)

    def turn_absolute(self, angle_deg: float):
        self._set_nav_curr_pose()
        delta_deg = angle_deg - self.curr_ang_deg_on_map
        actions_list = self.turn(delta_deg)
        self.recorded_actions_list.extend(actions_list)

    def face(self, name: str):
        self._set_nav_curr_pose()
        turn_right_angle = self.map.get_delta_angle_to(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.turn(turn_right_angle)

    def move_north(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_north_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_south(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_south_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_west(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_west_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_east(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_east_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_object(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_nearest_pos(self.curr_pos_on_map, name)
        self.move_to(pos)

    def move_forward(self, meters: float):
        self._set_nav_curr_pose()
        pos = self.map.get_forward_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, meters)
        self.move_to(pos)
