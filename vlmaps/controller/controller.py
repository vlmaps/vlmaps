from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig


class NavController:
    def __init__(self, controller_config: DictConfig):
        self.config = controller_config

    def convert_goal_to_actions(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[str]:
        return NotImplementedError

    def predict_poses_with_actions(
        self, start_pose: Tuple[float, float, float], actions_list: List[str]
    ) -> List[List[float]]:
        return NotImplementedError

    def convert_paths_to_actions(
        self, start_pose: Tuple[float, float, float], paths: List[Tuple[float, float]]
    ) -> List[List[str]]:
        return NotImplementedError

    def _normalize_angle(self, angle_deg: float):
        if angle_deg < -180:
            angle_deg += 360
        elif angle_deg > 180:
            angle_deg -= 360
        return angle_deg

    def _compute_dist(self, start_x: float, start_y: float, goal_x: float, goal_y: float):
        return np.sqrt((goal_x - start_x) * (goal_x - start_x) + (goal_y - start_y) * (goal_y - start_y))
