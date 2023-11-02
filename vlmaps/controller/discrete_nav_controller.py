from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from vlmaps.utils.mapping_utils import grid_id2base_pos_3d, base_pos2grid_id_3d

from .controller import NavController


class DiscreteNavController(NavController):
    """
    A controller that generates a list of actions consist of "move_foward", "turn_left", "turn_right"
    based on the start pose and goal positions on the full size map
    """

    def __init__(self, controller_config: DictConfig):
        super().__init__(controller_config)
        self.forward_dist = controller_config["forward_dist"]
        self.turn_angle_deg = controller_config["turn_angle"]
        self.gs = controller_config["gs"]
        self.cs = controller_config["cs"]

    def convert_goal_to_actions(self, start_pose: Tuple[float, float, float], goal: Tuple[float, float]) -> List[str]:
        """
        return discrete actions from start pose to goal position
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero for pointing upward, counter-clock-wise positive
        goal: (row, col) in full size map
        """
        actions_list = []
        curr_pos_x, curr_pos_y, curr_pos_z = grid_id2base_pos_3d(
            start_pose[0], start_pose[1], 0, self.cs, self.gs
        )  # x forward, y left, z upward
        curr_orientation_deg = start_pose[2]
        goal_x, goal_y, goal_z = grid_id2base_pos_3d(goal[0], goal[1], 0, self.cs, self.gs)  # up z, right x

        dx = goal_x - curr_pos_x
        dy = goal_y - curr_pos_y

        target_orientation = np.arctan2(dy, dx) * 180 / np.pi
        turn_right_angle = curr_orientation_deg - np.arctan2(dy, dx) * 180 / np.pi
        turn_right_angle = np.mod(turn_right_angle, 360)
        turn_right_angle = self._normalize_angle(turn_right_angle)

        dist = np.sqrt(dx * dx + dy * dy)

        turn_times = int(np.abs(np.round(turn_right_angle / self.turn_angle_deg)))
        turn_angle = np.sign(turn_right_angle) * turn_times * self.turn_angle_deg

        # update current orientation
        curr_orientation_deg = curr_orientation_deg - turn_angle
        curr_orientation_deg = self._normalize_angle(curr_orientation_deg)

        if turn_right_angle > 0:
            actions_list += ["turn_right"] * turn_times
        else:
            actions_list += ["turn_left"] * turn_times

        forward_times = int(np.abs(np.round(dist / self.forward_dist)))
        actions_list += ["move_forward"] * forward_times
        real_forward_dist = forward_times * self.forward_dist

        # update current position
        curr_pos_x = curr_pos_x + np.cos(curr_orientation_deg * np.pi / 180.0) * real_forward_dist
        curr_pos_y = curr_pos_y + np.sin(curr_orientation_deg * np.pi / 180.0) * real_forward_dist

        dist = self._compute_dist(curr_pos_x, curr_pos_y, goal_x, goal_y)

        # TODO: check later
        # if dist > self.config["goal_dist_thres"]:
        #     start_row, start_col, _ = base_pos2grid_id_3d(self.gs, self.cs, curr_pos_x, curr_pos_y, curr_pos_z)
        #     ext_actions_list = self.convert_goal_to_actions((start_row, start_col, curr_orientation_deg), goal)
        #     actions_list.extend(ext_actions_list)
        return actions_list

    def predict_poses_with_actions(
        self, start_pose: Tuple[float, float, float], actions_list: List[str]
    ) -> List[List[float]]:
        """
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero for pointing upward, counter-clock-wise positive
        predict a list of poses (x, z, angle_deg) (x right, z upward, angle_deg upward zero, clockwise) in simulator
        after executing each step of a list of actions
        """
        poses = []
        curr_pos_x, curr_pos_y, curr_pos_z = grid_id2base_pos_3d(start_pose[0], start_pose[1], 0, self.cs, self.gs)
        curr_angle_deg = start_pose[2]
        for action_i, action in enumerate(actions_list):
            if action == "move_forward":
                curr_pos_x = curr_pos_x + np.cos(np.deg2rad(curr_angle_deg)) * self.forward_dist
                curr_pos_y = curr_pos_y + np.sin(np.deg2rad(curr_angle_deg)) * self.forward_dist
                poses.append([curr_pos_x, curr_pos_y, curr_angle_deg])
            elif action == "turn_left":
                curr_angle_deg += self.turn_angle_deg
                poses.append([curr_pos_x, curr_pos_y, curr_angle_deg])
            elif action == "turn_right":
                curr_angle_deg -= self.turn_angle_deg
                poses.append([curr_pos_x, curr_pos_y, curr_angle_deg])

        return poses

    def convert_paths_to_actions(
        self, start_pose: Tuple[float, float, float], paths: List[List[float]]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero for pointing upward, counter-clock-wise positive
        goal: (row, col) in full size map
        Return discrete actions from start pose to goal position, and the predicted position after each step of action
        """
        actions_list = []
        poses_list = []
        curr_pose = start_pose
        for subgoal_i, subgoal in enumerate(paths):
            subgoal_actions_list = self.convert_goal_to_actions(curr_pose, subgoal)
            subgoal_poses_list = self.predict_poses_with_actions(curr_pose, subgoal_actions_list)
            actions_list.extend(subgoal_actions_list)
            poses_list.extend(subgoal_poses_list)

            # update current pose on map with the last predicted pose in simulation
            if len(subgoal_poses_list) > 0:
                curr_row, curr_col, _ = base_pos2grid_id_3d(
                    self.gs, self.cs, subgoal_poses_list[-1][0], subgoal_poses_list[-1][1], 0
                )
                curr_angle_deg = subgoal_poses_list[-1][2]
                curr_pose = (curr_row, curr_col, curr_angle_deg)

        return actions_list, poses_list
