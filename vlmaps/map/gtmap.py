from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation
from utils.ai2thor_constant import ai2thor_class_list
from utils.clip_mapping_utils import load_map
from utils.map.map import Map
from utils.planning_utils import find_similar_category_id, get_segment_islands_pos, mp3dcat


class GTMap(Map):
    def __init__(self, map_dir: str, map_config: DictConfig):
        super().__init__(map_dir, map_config)
        map_path = os.path.join(map_dir, "grid_gt_1.npy")
        self.map = load_map(map_path)
        self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        self.load_categories()
        self.obstacles_new_cropped = self.obstacles_cropped.copy()
        floor_mask = self.map_cropped == 2
        self.obstacles_new_cropped[floor_mask] = 1
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            map_config["dilate_iter"],
            map_config["gaussian_sigma"],
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0
        print("a GTMap is created")

    def load_categories(self, categories: List[str] = None):
        if categories is None:
            if self.map_config["categories"] == "mp3d":
                categories = mp3dcat.copy()
            elif self.map_config["categories"] == "ai2thor":
                categories = ai2thor_class_list.copy()
        self.categories = categories

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        cat_id = find_similar_category_id(name, self.categories)
        self.cropped_segment = self.map_cropped == cat_id
        forground = binary_closing(self.cropped_segment, iterations=3)
        forground = np.logical_and(forground, self.obstacles_cropped == 0)
        # cv2.imshow(name, (forground * 255).astype(np.uint8))
        # cv2.waitKey()
        self.cropped_segment = np.ones_like(self.cropped_segment, dtype=np.int32)
        self.cropped_segment[forground] = 0
        self.cropped_obstacles = self.obstacles[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        # visualize_segment_prediction(self.cropped_segment, [name, "other"], self.cropped_obstacles)

        contours, centers, bbox_list, _ = get_segment_islands_pos(self.cropped_segment, 0)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.xmin
            centers[i][1] += self.ymin
            bbox_list[i][0] += self.xmin
            bbox_list[i][1] += self.xmin
            bbox_list[i][2] += self.ymin
            bbox_list[i][3] += self.ymin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.xmin
                contours[i][j, 1] += self.ymin

        return contours, centers, bbox_list
