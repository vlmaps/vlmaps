from pathlib import Path
import yaml
import numpy as np
import h5py
import cv2

import torch

import matplotlib.patches as mpatches

# function to display the topdown map
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import List, Dict, Tuple, Set, Union


def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf


def load_ai2thor_pose(pose_filepath):
    with open(pose_filepath, "r") as f:
        line = f.readline()
        row = [float(x) for x in line.split()]
    return np.array(row, dtype=float).reshape((4, 4))


def load_real_world_poses(pose_filepath):
    ids_list = []
    tf_list = []
    with open(pose_filepath, "r") as f:
        line = f.readline()
        for line in f:
            row = [float(x) for x in line.strip().split()]
            id = int(row[-1])
            timestamp = row[0]
            pos = np.array(row[1:4])
            quat_xyzw = np.array(row[4:8])
            r = R.from_quat(quat_xyzw)
            rot_mat = r.as_matrix()
            tf = np.eye(4)
            tf[:3, :3] = rot_mat
            tf[:3, 3] = pos
            tf_list.append(tf)
            ids_list.append(id)
    return tf_list, ids_list


def load_tf_file(pose_filepath):
    with open(pose_filepath, "r") as f:
        line = f.readline()
        tf = np.array([float(x) for x in line.strip("\n").split()]).reshape((4, 4))
    return tf


def load_calib(calib_path):
    with open(calib_path, "r") as f:
        f.readline()
        f.readline()
        data = yaml.load(f, Loader=yaml.Loader)
    array = data["camera_matrix"]["data"]
    print("calib array", array)
    cam_mat = np.array([float(x) for x in array], dtype=np.float32).reshape((3, 3))
    return cam_mat


def load_pose(pose_filepath):
    with open(pose_filepath, "r") as f:
        line = f.readline()
        row = [float(x) for x in line.split()]
        pos = np.array(row[:3], dtype=float).reshape((3, 1))
        quat = row[3:]
        r = R.from_quat(quat)
        rot = r.as_matrix()

        return pos, rot


def load_depth_npy(depth_filepath: Union[Path, str]):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth


def load_depth_img(depth_filepath):
    return cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)


def load_semantic_npy(semantic_filepath):
    with open(semantic_filepath, "rb") as f:
        semantic = np.load(f)
    return semantic


def rob_pose2_cam_pose(pos, rot, camera_height):
    """
    Return homogeneous camera pose.
    Robot coordinate: z backward, y upward, x to the right
    Camera coordinate: z forward, x to the right, y downward
    And camera coordinate is camera_height meter above robot coordinate
    """

    rot_ro_cam = np.eye(3)
    rot_ro_cam[1, 1] = -1
    rot_ro_cam[2, 2] = -1
    rot = rot @ rot_ro_cam
    pos[1] += camera_height
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos.reshape(-1)

    return pose


def get_id2cls(obj2cls: dict):
    id2cls = {id: name for k, (id, name) in obj2cls.items()}
    return id2cls


def cvt_obj_id_2_cls_id(semantic: np.array, obj2cls: dict):
    h, w = semantic.shape
    semantic = semantic.flatten()
    u, inv = np.unique(semantic, return_inverse=True)
    return np.array([obj2cls[x][0] for x in u])[inv].reshape((h, w))


def load_lseg_feat(feat_filepath):
    with h5py.File(feat_filepath, "r") as f:
        feat = np.array(f["pixfeat"])
    return feat


def resize_feat(feat, h, w):
    """
    Input: feat (B, F, H, W). B is batch size, F is feature dimension, H, W are height and width
    """
    # b, f, _, _ = feat.shape
    # feat = np.resize(feat, (b, f, h, w))
    feat = torch.tensor(feat)
    feat = torch.nn.functional.interpolate(feat, (h, w), **{"mode": "bilinear", "align_corners": True})
    feat = feat.numpy()

    return feat


def depth2pc_ai2thor(depth, clipping_dist=0.1, fov=90):
    """
    Return 3xN array
    """

    h, w = depth.shape

    cam_mat = get_sim_cam_mat_with_fov(h, w, fov)

    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # x = x[int(h/2)].reshape((1, -1))
    # y = y[int(h/2)].reshape((1, -1))
    # z = depth[int(h/2)].reshape((1, -1))

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    # z = depth.reshape((1, -1))[:, :] + clipping_dist
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > 0.1
    mask2 = pc[2, :] < 10
    mask = np.logical_and(mask, mask2)
    # pc = pc[:, mask]
    return pc, mask


def depth2pc_real_world(depth, cam_mat):
    """
    Return 3xN array
    """

    h, w = depth.shape
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # x = x[int(h/2)].reshape((1, -1))
    # y = y[int(h/2)].reshape((1, -1))
    # z = depth[int(h/2)].reshape((1, -1))

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask_1 = pc[2, :] > 0.1
    # mask = mask_1
    mask_2 = pc[2, :] < 4
    mask = np.logical_and(mask_1, mask_2)
    # pc = pc[:, mask]
    return pc, mask


def rgb2pc(rgb):
    """
    Return 3xN int array
    """
    h, w, _ = rgb.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = x.reshape((1, -1))[:, :].flatten()
    y = y.reshape((1, -1))[:, :].flatten()
    rgb_pc = np.zeros((3, h * w), dtype=int)
    rgb_pc[:3, :] = rgb[y, x, :].T
    return rgb_pc


def depth2pc(depth, fov=90, intr_mat=None, min_depth=0.1, max_depth=10):
    """
    Return 3xN array and the mask of valid points in [min_depth, max_depth]
    """

    h, w = depth.shape

    cam_mat = intr_mat
    if intr_mat is None:
        cam_mat = get_sim_cam_mat_with_fov(h, w, fov)
    # cam_mat[:2, 2] = 0
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = x.reshape((1, -1))[:, :] + 0.5
    y = y.reshape((1, -1))[:, :] + 0.5
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > min_depth

    mask = np.logical_and(mask, pc[2, :] < max_depth)
    # pc = pc[:, mask]
    return pc, mask


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    # hsv_step = int(179 / n)
    # for j in range(0, n):
    #     hsv = np.array([hsv_step * j, 255, 255], dtype=np.uint8).reshape((1,1,3))
    #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #     rgb = rgb.reshape(-1)
    #     pallete[j * 3 + 0] = rgb[0]
    #     pallete[j * 3 + 1] = rgb[1]
    #     pallete[j * 3 + 2] = rgb[2]

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=[]):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


def transform_pc(pc, pose):
    """
    pose: the pose of the camera coordinate where the pc is in
    """
    # pose_inv = np.linalg.inv(pose)

    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])

    pc_global_homo = pose @ pc_homo

    return pc_global_homo[:3, :]


def pos2grid_id(gs, cs, xx, yy):
    x = int(gs / 2 + int(xx / cs))
    y = int(gs / 2 - int(yy / cs))
    return [x, y]


def grid_id2pos(gs, cs, x, y):
    xx = (x - gs / 2) * cs
    zz = (gs / 2 - y) * cs

    return xx, zz


def pos2grid_id_3d(gs, cs, camera_height, x_cam, y_cam, z_cam):
    x = int(gs / 2 + int(x_cam / cs))
    y = int(gs / 2 - int(z_cam / cs))
    z = int(camera_height / cs - y_cam / cs)
    return [x, y, z]


def grid_id2pos_3d(row, col, height, camera_height, cs, gs):
    cam_x = (col - gs / 2) * cs
    cam_z = (gs / 2 - row) * cs
    cam_y = camera_height - height * cs
    return [cam_x, cam_y, cam_z]


def base_pos2grid_id_3d(gs, cs, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]


def base_pos2grid_id_3d_batch(gs, cs, pos_base_np):
    """
    pos_base_np: [N, 3] np.int32
    """
    row = (gs / 2 - (pos_base_np[:, 0] / cs)).astype(np.int32).reshape((-1, 1))
    col = (gs / 2 - (pos_base_np[:, 1] / cs)).astype(np.int32).reshape((-1, 1))
    h = (pos_base_np[:, 2] / cs).astype(np.int32).reshape((-1, 1))
    return [row, col, h]


def grid_id2base_pos_3d(row, col, height, cs, gs):
    base_x = (gs / 2 - row) * cs
    base_y = (gs / 2 - col) * cs
    base_z = height * cs
    return [base_x, base_y, base_z]


def grid_id2base_pos_3d_batch(pos_grid_np, cs, gs):
    """
    pos_grid_np: [N, 3] np.int32
    """
    base_x = (gs / 2 - pos_grid_np[:, 0]) * cs
    base_y = (gs / 2 - pos_grid_np[:, 1]) * cs
    base_z = pos_grid_np[:, 2] * cs
    return [base_x, base_y, base_z]


def base_rot_mat2theta(rot_mat: np.ndarray) -> float:
    """Convert base rotation matrix to rotation angle (rad) assuming x is forward, y is left, z is up

    Args:
        rot_mat (np.ndarray): (3,3) rotation matrix

    Returns:
        float: rotation angle
    """
    theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return theta


def grid_id2pos_3d_batch(pos_np: np.ndarray, camera_height: float, cs: float, gs: int):
    """
    pos_np: [N, 3] np.int32
    """
    cam_x = (pos_np[:, 1] - gs / 2) * cs
    cam_z = (gs / 2 - pos_np[:, 0]) * cs
    cam_y = camera_height - pos_np[:, 2] * cs

    return np.concatenate([cam_x.reshape((-1, 1)), cam_y.reshape((-1, 1)), cam_z.reshape((-1, 1))], axis=1)


def get_vfov(hfov, h, w):
    vfov = hfov * h / w
    return vfov


def get_frustum_4pts(dmin, dmax, theta, hf_2, vf_2):
    theta -= np.pi / 2
    theta = -theta
    tan_theta_hf_2 = np.tan(theta + hf_2)
    tmp = 1.0 / (np.sin(theta) * tan_theta_hf_2 + np.cos(theta))
    x1 = dmin * tmp
    y1 = tan_theta_hf_2 * x1

    x4 = dmax * tmp
    y4 = tan_theta_hf_2 * x4

    tan_theta_min_hf_2 = np.tan(theta - hf_2)
    tmp2 = 1.0 / (np.sin(theta) * tan_theta_min_hf_2 + np.cos(theta))

    x2 = dmin * tmp2
    y2 = tan_theta_min_hf_2 * x2

    x3 = dmax * tmp2
    y3 = tan_theta_min_hf_2 * x3

    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float64)


def generate_mask(gs, cs, hfov, theta, depth, robot_x, robot_y):
    """
    Generate mask based on viewing lines to the maximal range in that angle (max of the column of the depth map)
    """
    mask = np.zeros((gs, gs), dtype=np.int8)
    sp = pos2grid_id(gs, cs, robot_x, robot_y)

    # get the depth value of end points
    z = np.max(depth, axis=0)

    # get the angle of the end points
    w = depth.shape[1]
    inc = hfov / float(w)
    angles = theta + hfov / 2.0 - np.arange(w) * hfov / w

    # get the list of end points positions
    x = robot_x + z * np.cos(angles)
    y = robot_y + z * np.sin(angles)

    for i in range(w):
        ep = pos2grid_id(gs, cs, x[i], y[i])
        cv2.line(mask, sp, ep, 255)
        mask[ep[0], ep[1]] = 0
    return mask


def save_map(save_path, map):
    with open(save_path, "wb") as f:
        np.save(f, map)
        print(f"{save_path} is saved.")


def load_map(load_path):
    with open(load_path, "rb") as f:
        map = np.load(f)
    return map


def save_3d_map(
    save_path: str,
    grid_feat: np.ndarray,
    grid_pos: np.ndarray,
    weight: np.ndarray,
    occupied_ids: np.ndarray,
    mapped_iter_list: Set[int],
    grid_rgb: np.ndarray = None,
) -> None:
    """Save 3D voxel map with features

    Args:
        save_path (str): path to save the map as an H5DF file.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) the position of the occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mapped_iter_list", data=np.array(mapped_iter_list, dtype=np.int32))
        f.create_dataset("grid_feat", data=grid_feat)
        f.create_dataset("grid_pos", data=grid_pos)
        f.create_dataset("weight", data=weight)
        f.create_dataset("occupied_ids", data=occupied_ids)
        if grid_rgb is not None:
            f.create_dataset("grid_rgb", data=grid_rgb)


def load_3d_map(map_path: str) -> Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D voxel map with features

    Args:
        map_path (str): path to save the map as an H5DF file.
    Return:
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) each row is the (row, col, height) of an occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]
        return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb


d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


def get_sim_cam_mat(h, w):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / 2.0
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def project_point(cam_mat, p):
    new_p = cam_mat @ p.reshape((3, 1))
    z = new_p[2, 0]
    new_p = new_p / new_p[2, 0]
    x = int(new_p[0, 0] - 0.5)
    y = int(new_p[1, 0] - 0.5)
    return x, y, z


def project_points(cam_mat, p):
    new_p = cam_mat @ p.reshape((3, -1))
    z = new_p[2, :]
    new_p = new_p / new_p[2, :]
    x = (new_p[0, :] - 0.5).astype(int)
    y = (new_p[1, :] - 0.5).astype(int)
    return x, y, z


def get_sim_cam_mat_with_fov(h, w, fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def load_obj2cls_dict(filepath):
    obj2cls_dict = dict()
    with open(filepath, "r") as f:
        for line in f:
            row = line.split(":")
            obj_id = int(row[0])
            cls_id = int(row[1].split(",")[0].strip())
            cls_name = row[1].split(",")[1].strip()
            obj2cls_dict[obj_id] = (cls_id, cls_name)
    return obj2cls_dict


def save_clip_sparse_map(save_path: str, clip_sparse_map: np.ndarray, robot_pose_list: List[np.ndarray]):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("clip_sparse_map", data=clip_sparse_map)
        f.create_dataset("robot_pose_list", data=robot_pose_list)


def load_clip_sparse_map(load_path: str):
    with h5py.File(load_path, "r") as f:
        clip_sparse_map = f["clip_sparse_map"][:]
        robot_pose_list = f["robot_pose_list"][:]
    return clip_sparse_map, robot_pose_list


def load_calib(calib_file: str) -> np.ndarray:
    """Load calibration file."""
    with open(calib_file, "r") as f:
        line = f.readline()
        calib = np.array([float(x) for x in line.strip().split(",")]).reshape((3, 3))
    return calib


def load_real_pose(pose_file: str) -> np.ndarray:
    """Load real pose file."""
    with open(pose_file, "r") as f:
        line = f.readline()
        pose = np.array([float(x) for x in line.strip().split("\t")]).reshape((4, 4))
    return pose


def load_real_depth(depth_file: str) -> np.ndarray:
    """Load real depth file."""
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    return depth


def pos2grid_id_3d_real(gs, cs, camera_height, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    height = int(camera_height / cs - z_base / cs)
    return [row, col, height]


def grid_id2pos_3d_real(gs, cs, camera_height, row, col, height):
    x_base = (gs / 2 - row) * cs
    y_base = (gs / 2 - col) * cs
    z_base = (camera_height / cs - height) * cs
    return [x_base, y_base, z_base]


def grid_id2pos_3d_real_batch(pos_np: np.ndarray, camera_height: float, cs: float, gs: int):
    """
    pos_np: [N, 3] np.int32
    """
    base_x = (gs / 2 - pos_np[:, 0]) * cs
    base_y = (gs / 2 - pos_np[:, 1]) * cs
    base_z = camera_height - pos_np[:, 2] * cs

    return np.concatenate([base_x.reshape((-1, 1)), base_y.reshape((-1, 1)), base_z.reshape((-1, 1))], axis=1)
