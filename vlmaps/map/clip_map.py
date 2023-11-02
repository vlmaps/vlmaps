from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import clip
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import binary_closing, binary_dilation
import torch
from utils.ai2thor_constant import ai2thor_class_list
from utils.clip_mapping_utils import load_map
from utils.clip_utils import get_text_feats
from utils.map.map import Map
from utils.planning_utils import find_similar_category_id, get_segment_islands_pos, mp3dcat, multiple_templates


class CLIPMap(Map):
    def __init__(self, map_dir: str, map_config: DictConfig):
        super().__init__(map_config)
        map_path = os.path.join(map_dir, "grid_clip_1.npy")
        self.map = load_map(map_path)
        self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        self._init_clip()
        self.load_categories()
        print("a CLIPMap is created")

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

    def load_categories(self, categories: List[str] = None):
        if categories is None:
            if self.map_config["categories"] == "mp3d":
                categories = mp3dcat.copy()
            elif self.map_config["categories"] == "ai2thor":
                categories = ai2thor_class_list.copy()
        self.categories = categories

        mul_tmp = multiple_templates.copy()
        multi_temp_landmarks_other = [x.format(lm) for lm in categories for x in mul_tmp]
        text_feats = get_text_feats(multi_temp_landmarks_other, self.clip_model, self.clip_feat_dim)

        # average the features
        text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
        text_feats = np.mean(text_feats, axis=1)

        # text_feats = get_text_feats(landmarks, self.clip_model, self.clip_feat_dim)
        map_feats = self.map_cropped.reshape((-1, self.map_cropped.shape[-1]))
        scores_list = map_feats @ text_feats.T
        scores_masks = scores_list.reshape((self.map_cropped.shape[0], self.map_cropped.shape[1], -1))
        scores_masks = scores_masks / np.max(scores_masks.reshape((-1, scores_masks.shape[2])), axis=0).reshape(
            (1, 1, scores_masks.shape[2])
        )
        # threshold = np.percentile(scores_masks, 95, axis=)
        valid_masks = []
        for i in range(scores_masks.shape[2]):
            threshold = np.percentile(scores_masks[:, :, i], 95)
            valid_masks.append(scores_masks[:, :, i] > threshold)

        self.labeled_map_cropped = valid_masks
        return valid_masks

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        cat_id = find_similar_category_id(name, self.categories)
        forground = binary_closing(self.labeled_map_cropped[cat_id], iterations=3)
        forground = np.logical_and(forground, self.obstacles_cropped == 0)

        contours, centers, bbox_list, _ = get_segment_islands_pos(forground, 1)

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
