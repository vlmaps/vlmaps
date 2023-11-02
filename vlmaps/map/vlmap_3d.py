from __future__ import annotations
import os
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_closing
from omegaconf import DictConfig, OmegaConf
import torch
import clip

from utils.clip_mapping_utils import load_map
from utils.planning_utils import (
    mp3dcat,
    get_dynamic_obstacles_map_3d,
    get_lseg_score,
    segment_lseg_map,
    find_similar_category_id,
    get_segment_islands_pos,
)
from utils.ai2thor_constant import ai2thor_class_list
from utils.map.map_3d import Map3D
from utils.map.map import Map
from utils.concept_fusion_utils import generate_fused_features, load_concept_fusion_map

from typing import List, Dict, Tuple, Any


class VLMap3D(Map3D):
    def __init__(self, map_dir: str, map_config: DictConfig, map_3d_dir: str):
        super().__init__(map_dir, map_config)

        self.map_3d_dir = map_3d_dir
        self.map_3d_path = os.path.join(map_3d_dir, "vlmaps_lseg_3d.h5df")
        self.load_map(self.map_3d_path)
        self._init_clip(clip_version="ViT-B/32")
        self._customize_obstacle_map(
            map_config["potential_obstacle_names"],
            map_config["obstacle_names"],
            vis=False,
        )
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            map_config["dilate_iter"],
            map_config["gaussian_sigma"],
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0
        self.categories = None
        self.scores_mat = None
    
    def init_categories(self, categories: List[str]):
        self.categories = categories
        self.scores_mat = get_lseg_score(self.clip_model, categories, self.grid_feat, self.clip_feat_dim, use_multiple_templates=True, add_other = True) # score for name and other

    def load_map(self, map_path: str):
        mapped_iter_list, self.grid_feat, self.grid_pos, self.weight, self.occupied_ids, self.grid_rgb = load_concept_fusion_map(map_path)

    def _customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            potential_obstacle_names,
            obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.xmin,
            self.ymin,
            self.clip_feat_dim,
            vis=vis,
        )
    
    def get_predict_mask(self, name: str) -> np.ndarray:
        if self.scores_mat is not None and self.categories is not None:
            id = find_similar_category_id(name, self.categories)
            ids = (np.argmax(self.scores_mat, axis=1) == id).flatten()
            mask = np.zeros_like(self.obstacles_cropped)
            mask[self.grid_pos[ids, 0] - self.xmin, self.grid_pos[ids, 1] - self.ymin] = 1
            return mask
        return get_lseg_score(self.clip_model, [name], self.grid_feat, self.clip_feat_dim, use_multiple_templates=True, add_other = True).flatten() # score for name and other

    def get_pos(self, name: str) -> Tuple[float, float]:
        id = find_similar_category_id(name, self.categories)
        ids = (np.argmax(self.scores_mat, axis=1) == id).flatten()
        # scores_list = get_lseg_score(self.clip_model, [name], self.grid_feat, self.clip_feat_dim, use_multiple_templates=True, add_other = True) # score for name and other
        scores_map = np.zeros_like(self.obstacles_cropped, dtype=np.int32)
        scores_map[self.grid_pos[ids, 0] - self.xmin, self.grid_pos[ids, 1] - self.ymin] = 1

        contours, centers, bbox_list, _ = get_segment_islands_pos(scores_map, 1)
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
    