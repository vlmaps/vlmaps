from typing import Dict, List, Tuple

import habitat_sim
import numpy as np
from omegaconf import DictConfig

from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from vlmaps.utils.habitat_utils import agent_state2tf


class HabitatTask:
    def __init__(self, config: DictConfig):
        self.config = config
        self.goals: List[List[Tuple[float, float]]]
        self.goals = []

    def setup_scene(self, vlmaps_dataloader: VLMapsDataloaderHabitat):
        self.vlmaps_dataloader = vlmaps_dataloader

    def load_task(self):
        """
        Setup goal positions for each goal
        """
        return NotImplementedError

    def reset_metrics(self):
        self.n_tot_tasks = 0
        self.n_success_tasks = 0
        self.n_tot_subgoals = 0
        self.n_success_subgoals = 0

    def test_actions(
        self,
        sim: habitat_sim.Simulator,
        init_agent_state: habitat_sim.AgentState,
        actions_list: List[str],
        goals_positions: List[List[List[float]]],
    ) -> List[bool]:
        sim.get_agent(0).set_state(init_agent_state)
        actions_set = {"move_forward", "turn_left", "turn_right"}
        stop_list = []
        success_list = [False] * len(self.goals)
        min_dist_list = [-1] * len(self.goals)
        for action_i, action in enumerate(actions_list):
            if action == "stop":
                final_state = sim.get_agent(0).get_state()
                goal_id = len(stop_list)
                stop_list.append("stop")
                hab_tf = agent_state2tf(final_state)
                row, col, angle_deg = self.vlmaps_dataloader.convert_habitat_tf_to_full_map_pose(hab_tf)
                goal_positions = self.goals[goal_id]
                success, min_dist = self._check_reached_goal_positions((row, col), goal_positions)
                success_list[goal_id] = success
                min_dist_list[goal_id] = min_dist
                continue

            if action in actions_set:
                sim.step(action)
                continue

        return success_list, min_dist_list

    def _check_min_dist_to_goal_positions(
        self, checked_pos: Tuple[float, float], goal_positions: List[Tuple[float, float]]
    ) -> float:
        row, col = checked_pos
        min_dist_pix = np.inf
        for pos in goal_positions:
            goal_row, goal_col = pos
            drow, dcol = goal_row - row, goal_col - col
            dist = np.sqrt(drow * drow + dcol * dcol)
            min_dist_pix = dist if dist < min_dist_pix else min_dist_pix
        return min_dist_pix * self.vlmaps_dataloader.cs

    def _check_reached_goal_positions(
        self, checked_pos: Tuple[float, float], goal_positions: List[Tuple[float, float]]
    ) -> Tuple[bool, float]:
        min_dist = self._check_min_dist_to_goal_positions(checked_pos, goal_positions)
        if min_dist < self.config["nav"]["valid_range"]:
            return True, min_dist
        return False, min_dist

    def _check_min_dist_to_goal_tfs(self, checked_tf: np.ndarray, goal_tfs: List[np.ndarray]) -> float:
        min_dist = np.inf
        pos = checked_tf[:3, 3]
        for goal_tf in goal_tfs:
            goal_pos = goal_tf[:3, 3]
            dist = np.linalg.norm(goal_pos - pos)
            min_dist = dist if dist < min_dist else min_dist
        return min_dist

    def _check_reached_goal_tfs(self, checked_tf: np.ndarray, goal_tfs: List[np.ndarray]) -> Tuple[bool, float]:
        min_dist = self._check_min_dist_to_goal_tfs(checked_tf, goal_tfs)
        if min_dist < self.config["nav"]["valid_range"]:
            return True, min_dist
        return False, min_dist
