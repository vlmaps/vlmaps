import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
import h5py

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    load_3d_map,
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_img,
    load_depth_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet


def visualize_pc(pc: np.ndarray):
    """Take (N, 3) point cloud and visualize it using open3d.

    Args:
        pc (np.ndarray): (N, 3) point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


class VLMapBuilderCam:
    def __init__(
        self,
        data_dir: Path,
        map_config: DictConfig,
        pose_path: Path,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        base2cam_tf: np.ndarray,
        base_transform: np.ndarray,
    ):
        self.data_dir = data_dir
        self.pose_path = pose_path
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.map_config = map_config
        self.base2cam_tf = base2cam_tf
        self.base_transform = base_transform
        self.rot_type = map_config.pose_info.rot_type

    def create_camera_map(self):
        """
        build a map centering at the global original frame. The poses are camera pose in the global coordinate frame.
        """
        # access config info
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        depth_sample_rate = self.map_config.depth_sample_rate
        self.camera_pose_tfs = np.loadtxt(self.pose_path)
        if self.rot_type == "quat":
            self.camera_pose_tfs = [cvt_pose_vec2tf(x) for x in self.camera_pose_tfs]
        elif self.rot_type == "mat":
            self.camera_pose_tfs = [x.reshape((4, 4)) for x in self.camera_pose_tfs]
        else:
            raise ValueError("Invalid rotation type")

        self.map_save_dir = self.data_dir / "vlmap_cam"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps_cam.h5df"

        # init lseg model
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        # load camera calib matrix in config
        calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))

        # get global pcd
        pbar = tqdm(
            zip(self.rgb_paths, self.depth_paths, self.camera_pose_tfs),
            total=len(self.rgb_paths),
            desc="Get Global Map",
        )
        global_pcd = o3d.geometry.PointCloud()
        for frame_i, (rgb_path, depth_path, camera_pose_tf) in enumerate(pbar):
            if frame_i % self.map_config.skip_frame != 0:
                continue
            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path.as_posix())
            pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=10)
            transform_tf = camera_pose_tf  # @ self.habitat2cam_rot_tf
            pc_global = transform_pc(pc, transform_tf)  # (3, N)
            pcd_global = o3d.geometry.PointCloud()
            pcd_global.points = o3d.utility.Vector3dVector(pc_global.T)
            global_pcd += pcd_global
            # downsample global_pcd
            # if frame_i % (100 * self.map_config.skip_frame) == 99:
            #     global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)

            # if frame_i % 50 == 0:
            #     o3d.visualization.draw_geometries([global_pcd])

        self.pcd_min = np.min(np.asarray(global_pcd.points), axis=0)
        self.pcd_max = np.max(np.asarray(global_pcd.points), axis=0)

        grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id = self._init_map(
            self.pcd_min, self.pcd_max, cs, self.map_save_path
        )

        cv_map = np.zeros((self.grid_size[0], self.grid_size[2], 3), dtype=np.uint8)
        height_map = -100 * np.ones(self.grid_size[[0, 2]], dtype=np.float32)

        pbar = tqdm(zip(self.rgb_paths, self.depth_paths, self.camera_pose_tfs), total=len(self.rgb_paths))
        for frame_i, (rgb_path, depth_path, camera_pose_tf) in enumerate(pbar):
            if frame_i % self.map_config.skip_frame != 0:
                continue
            if frame_i in mapped_iter_set:
                continue
            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path.as_posix())

            # # get pixel-aligned LSeg features
            pix_feats = get_lseg_feat(
                lseg_model, rgb, ["example"], lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std
            )
            pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

            # backproject depth point cloud
            pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=100)

            # transform the point cloud to global frame (init base frame)
            transform_tf = camera_pose_tf  # @ self.habitat2cam_rot_tf
            pc_global = transform_pc(pc, transform_tf)  # (3, N)

            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                row, height, col = np.round(((p - self.pcd_min) / cs)).astype(int)

                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                px, py, pz = project_point(pix_feats_intr, p_local)
                if row >= height_map.shape[0] or col >= height_map.shape[1] or row < 0 or col < 0:
                    print("out of range")
                    continue

                if height > height_map[row, col]:
                    height_map[row, col] = height
                    cv_map[row, col, :] = rgb_v

                # when the max_id exceeds the reserved size,
                # double the grid_feat, grid_pos, weight, grid_rgb lengths
                if max_id >= grid_feat.shape[0]:
                    grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(
                        grid_feat, grid_pos, weight, grid_rgb
                    )

                # apply the distance weighting according to
                # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
                radial_dist_sq = np.sum(np.square(p_local))
                sigma_sq = 0.6
                alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

                # update map features
                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, height, col]
                    if occupied_id == -1:
                        occupied_ids[row, height, col] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v
                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, height, col]
                        max_id += 1
                    else:
                        grid_feat[occupied_id] = (
                            grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                        ) / (weight[occupied_id] + alpha)
                        if weight[occupied_id] + alpha == 0:
                            print("weight is 0")
                        grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                            weight[occupied_id] + alpha
                        )
                        weight[occupied_id] += alpha

            mapped_iter_set.add(frame_i)
            if frame_i % (self.map_config.skip_frame * 100) == self.map_config.skip_frame * 99:
                print(f"Temporarily saving {max_id} features at iter {frame_i}...")
                self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)

        self.save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)

    def create_mobile_base_map(self):
        """
        TODO: To be implemented

        build the 3D map centering at the first base frame.
        """
        return NotImplementedError

    def _init_map(self, pcd_min: np.ndarray, pcd_max: np.ndarray, cs: float, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size grid_size = (pcd_max - pcd_min) / cs + 1
        """
        grid_size = np.ceil((pcd_max - pcd_min) / cs + 1).astype(int)  # col, height, row
        self.grid_size = grid_size
        occupied_ids = -1 * np.ones(grid_size[[0, 1, 2]], dtype=np.int32)
        grid_feat = np.zeros((grid_size[0] * grid_size[2], self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.int32)
        weight = np.zeros((grid_size[0] * grid_size[2]), dtype=np.float32)
        grid_rgb = np.zeros((grid_size[0] * grid_size[2], 3), dtype=np.uint8)
        # init the map related variables
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        max_id = 0

        # check if there is already saved map
        if os.path.exists(map_path):
            (mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs) = (
                self.load_3d_map(self.map_save_path)
            )
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]
            self.pcd_min = pcd_min

        return grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id

    @staticmethod
    def load_3d_map(map_path: Union[Path, str]):
        with h5py.File(map_path, "r") as f:
            mapped_iter_list = f["mapped_iter_list"][:].tolist()
            grid_feat = f["grid_feat"][:]
            grid_pos = f["grid_pos"][:]
            weight = f["weight"][:]
            occupied_ids = f["occupied_ids"][:]
            grid_rgb = f["grid_rgb"][:]
            pcd_min = f["pcd_min"][:]
            pcd_max = f["pcd_max"][:]
            cs = f["cs"][()]
            return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs

    def _init_lseg(self):
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc

    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb

    def _reserve_grid(
        self, row: int, col: int, height: int, gs: int, vh: int, init_height_id: int, occupied_ids: np.ndarray
    ) -> Tuple[int, int]:
        if not self._out_of_range(row, col, height, gs, vh):
            print("not out of range")
            return occupied_ids, init_height_id, vh
        if height < 0:
            tmp = -np.ones((gs, gs, -height), dtype=occupied_ids.dtype)
            print("smaller than 0")
            print("before: ", occupied_ids.shape)
            occupied_ids = np.concatenate([occupied_ids, tmp], axis=2)
            print("after: ", occupied_ids.shape)
            init_height_id += -height
            vh += -height
        elif height >= vh:
            tmp = -np.ones((gs, gs, height - vh + 1), dtype=occupied_ids.dtype)
            print("larger and equal than vh")
            print("before: ", occupied_ids.shape)
            occupied_ids = np.concatenate([tmp, occupied_ids], axis=2)
            print("after: ", occupied_ids.shape)
            init_height_id += height
            vh += height
        return occupied_ids, init_height_id, vh

    def save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        with h5py.File(self.map_save_path, "w") as f:
            f.create_dataset("mapped_iter_list", data=np.array(list(mapped_iter_set), dtype=np.int32))
            f.create_dataset("grid_feat", data=grid_feat)
            f.create_dataset("grid_pos", data=grid_pos)
            f.create_dataset("weight", data=weight)
            f.create_dataset("occupied_ids", data=occupied_ids)
            f.create_dataset("grid_rgb", data=grid_rgb)
            f.create_dataset("pcd_min", data=self.pcd_min)
            f.create_dataset("pcd_max", data=self.pcd_max)
            f.create_dataset("cs", data=self.map_config.cell_size)
