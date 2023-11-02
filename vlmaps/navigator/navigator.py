import numpy as np
import pyvisgraph as vg
from vlmaps.utils.navigation_utils import build_visgraph_with_obs_map, plan_to_pos_v2
from typing import Tuple, List, Dict


class Navigator:
    def __init__(self):
        pass

    def build_visgraph(self, obstacle_map: np.ndarray, rowmin: float, colmin: float, vis: bool = False):
        self.obs_map = obstacle_map
        self.visgraph = build_visgraph_with_obs_map(obstacle_map, vis=vis)
        self.rowmin = rowmin
        self.colmin = colmin

    def plan_to(
        self, start_full_map: Tuple[float, float], goal_full_map: Tuple[float, float], vis: bool = False
    ) -> List[List[float]]:
        """
        Take full map start (row, col) and full map goal (row, col) as input
        Return a list of full map path points (row, col) as the palnned path
        """
        start = self._convert_full_map_pos_to_cropped_map_pos(start_full_map)
        goal = self._convert_full_map_pos_to_cropped_map_pos(goal_full_map)
        if self._check_if_start_in_graph_obstacle(start):
            self._rebuild_visgraph(start, vis)
        paths = plan_to_pos_v2(start, goal, self.obs_map, self.visgraph, vis)
        paths = self.shift_path(paths, self.rowmin, self.colmin)
        return paths

    def shift_path(self, paths: List[List[float]], row_shift: int, col_shift: int) -> List[List[float]]:
        shifted_paths = []
        for point in paths:
            shifted_paths.append([point[0] + row_shift, point[1] + col_shift])
        return shifted_paths

    def _check_if_start_in_graph_obstacle(self, start: Tuple[float, float]):
        startvg = vg.Point(start[0], start[1])
        poly_id = self.visgraph.point_in_polygon(startvg)
        if poly_id != -1 and self.obs_map[int(start[0]), int(start[1])] == 1:
            return True
        return False

    def _rebuild_visgraph(self, start: Tuple[float, float], vis: bool = False):
        self.visgraph = build_visgraph_with_obs_map(
            self.obs_map, use_internal_contour=True, internal_point=start, vis=vis
        )

    def _convert_full_map_pos_to_cropped_map_pos(self, full_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        full_map_pos: (row, col) in full map
        Return (row, col) in cropped_map
        """
        print("full_map_pos: ", full_map_pos)
        print("self.rowmin: ", self.rowmin)
        print("self.colmin: ", self.colmin)
        return [full_map_pos[0] - self.rowmin, full_map_pos[1] - self.colmin]

    def _convert_cropped_map_pos_to_full_map_pos(self, cropped_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        cropped_map_pos: (row, col) in cropped map
        Return (row, col) in full map
        """
        return [cropped_map_pos[0] + self.rowmin, cropped_map_pos[1] + self.colmin]
