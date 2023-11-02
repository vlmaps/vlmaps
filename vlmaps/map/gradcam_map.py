from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import clip
import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch
from utils.ai2thor_constant import ai2thor_class_list
from utils.clip_mapping_utils import load_map
from utils.clip_utils import get_text_feats
from utils.map.map import Map
from utils.planning_utils import find_similar_category_id, get_segment_islands_pos, mp3dcat, multiple_templates


class GradCAMMap(Map):
    def __init__(self, map_dir: str, map_config: DictConfig):
        super().__init__(map_dir, map_config)
        map_path = os.path.join(map_dir, "grid_clip_gradcam_1.npy")
        self.map = load_map(map_path)
        self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        self._init_clip()
        self._customize_obstacle_map()
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped,
            map_config["dilate_iter"],
            map_config["gaussian_sigma"],
        )
        self.load_categories()

        print("a GradCAMMap is created")

    def _init_clip(self, clip_version="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def _customize_obstacle_map(
        self,
        vis: bool = False,
    ):
        """
        Remove floor categories from obstacle map
        """
        floor_mask = self.map_cropped[:, :, 2] > 0.75
        floor_mask = binary_closing(floor_mask, iterations=3)
        self.obstacles_new_cropped = self.obstacles_cropped.copy()
        self.obstacles_new_cropped[floor_mask] = 1  # 1 is free space

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
        foreground = self.map_cropped[:, :, cat_id] > 0.75
        foreground = binary_closing(foreground, iterations=3)
        foreground = np.logical_and(foreground, self.obstacles_new_cropped == 0)

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
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
