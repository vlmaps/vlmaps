import cv2
import h5py
import matplotlib.patches as mpatches
import numpy as np

# function to display the topdown map
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
import yaml

import examples.context as context

# from utils.utils import *

# from socratic_navigation import navigate_with_pose


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


def load_depth(depth_filepath):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth


def load_semantic(semantic_filepath):
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


def depth2pc(depth, fov=90):
    """
    Return 3xN array
    """

    h, w = depth.shape

    cam_mat = get_sim_cam_mat_with_fov(h, w, fov)
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > 0.1
    return pc, mask


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)

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
    return mask


def save_map(save_path, map):
    with open(save_path, "wb") as f:
        np.save(f, map)
        print(f"{save_path} is saved.")


def load_map(load_path):
    with open(load_path, "rb") as f:
        map = np.load(f)
    return map


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
    x = int(new_p[0, 0] + 0.5)
    y = int(new_p[1, 0] + 0.5)
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
