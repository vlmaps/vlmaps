from typing import Dict, List, Tuple

from controller import NavController
import numpy as np
from omegaconf import DictConfig
from vlmaps.utils.mapping_utils import grid_id2pos, pos2grid_id


class ContinuousNavController(NavController):
    """
    A controller that generates a list of actions consist of "move_foward", "turn_left", "turn_right"
    based on the start pose and goal positions on the full size map
    """

    def __init__(self, controller_config: DictConfig):
        super().__init__(controller_config)
        self.forward_vel = controller_config["forward_vel"]
        self.turn_vel_deg_per_s = controller_config["turn_vel_deg_per_s"]
        self.gs = controller_config["gs"]
        self.cs = controller_config["cs"]

    def convert_goal_to_actions(
        self, start_pose: Tuple[float, float, float], goal: Tuple[float, float]
    ) -> List[Tuple[str, float, float]]:
        """
        return continuous actions ("forward"/"turn", distance/angle, time)
        from start pose to goal position
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero
        for pointing upward, clock-wise positive
        goal: (row, col) in full size map
        """
        actions_list = []
        curr_pos_x, curr_pos_z = grid_id2pos(self.gs, self.cs, start_pose[1], start_pose[0])  # up z, right x
        curr_orientation_deg = start_pose[2]  # up is zero, clock-wise goal_x, goal_z = grid_id2pos(
        goal_x, goal_z = grid_id2pos(self.gs, self.cs, goal[1], goal[0])  # up z, right x

        dx = goal_x - curr_pos_x
        dz = goal_z - curr_pos_z

        goal_x, goal_z = grid_id2pos(self.gs, self.cs, goal[1], goal[0])  # up z, right x

        dx = goal_x - curr_pos[0]
        dz = goal_z - curr_pos[1]

        turn_right_angle = np.arctan2(dx, dz) * 180 / np.pi - curr_orientation_deg
        turn_right_angle = np.mod(turn_right_angle, 360)
        turn_right_angle = self._normalize_angle(turn_right_angle)

        dist = np.sqrt(dx * dx + dz * dz)

        turn_times = np.abs(turn_right_angle / self.turn_vel_deg_per_s)
        turn_angle = (
            np.sign(turn_right_angle) * turn_times * self.turn_vel_deg_per_s
        )  # for real robot, turn left is positive

        # update current orientation
        curr_orientation_deg = curr_orientation_deg + turn_angle
        curr_orientation_deg = self._normalize_angle(curr_orientation_deg)

        if turn_times > 1e-6:
            actions_list += [("turn", np.deg2rad(-turn_angle), turn_times)]

        forward_times = np.abs(dist / self.forward_vel)
        real_forward_dist = forward_times * self.forward_vel
        if forward_times > 1e-6:
            actions_list += [("forward", real_forward_dist, forward_times)]
        curr_x = curr_pos[0] + np.sin(curr_orientation_deg * np.pi / 180.0) * real_forward_dist
        curr_z = curr_pos[1] + np.cos(curr_orientation_deg * np.pi / 180.0) * real_forward_dist
        curr_pos = [curr_x, curr_z]
        return actions_list

    def predict_poses_with_actions(
        self,
        start_pose: Tuple[float, float, float],
        actions_list: List[Tuple[str, float, float]],
    ) -> List[List[float]]:
        """
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero for pointing upward, clock-wise positive
        predict a list of poses (x, z, angle_deg) (x right, z upward, angle_deg upward zero, clockwise) in simulator
        after executing each step of a list of actions
        """
        poses = []
        curr_pos_x, curr_pos_z = grid_id2pos(self.gs, self.cs, start_pose[1], start_pose[0])
        curr_angle_deg = start_pose[2]
        for action_i, (action, delta, times) in enumerate(actions_list):
            if action == "forward":
                curr_pos_x = curr_pos_x + np.sin(np.deg2rad(curr_angle_deg)) * delta
                curr_pos_z = curr_pos_z + np.cos(np.deg2rad(curr_angle_deg)) * delta
                poses.append([curr_pos_x, curr_pos_z, curr_angle_deg])
            elif action == "turn":
                curr_angle_deg -= delta
                poses.append([curr_pos_x, curr_pos_z, curr_angle_deg])

        return poses

    def convert_paths_to_actions(
        self, start_pose: Tuple[float, float, float], paths: List[List[float]]
    ) -> Tuple[List[Tuple[str, float, float]], List[List[float]]]:
        """
        start_pose: (row, col, angle_deg) in full size map, angle_deg is zero for pointing upward, clock-wise positive
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
            curr_col, curr_row = pos2grid_id(self.gs, self.cs, subgoal_poses_list[-1][0], subgoal_poses_list[-1][1])
            curr_angle_deg = subgoal_poses_list[-1][2]
            curr_pose = (curr_row, curr_col, curr_angle_deg)

        return actions_list, poses_list
