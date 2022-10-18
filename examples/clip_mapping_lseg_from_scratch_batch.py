import examples.context as context
import time
import os
import argparse
import math

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import clip

from matplotlib import pyplot as plt

from utils.clip_mapping_utils import *
from utils.time_utils import Tic

from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image

def create_lseg_map_batch(img_save_dir, camera_height, cs=0.05, gs=1000, depth_sample_rate=100):
    hfov_deg = 90
    hfov_rad = hfov_deg * np.pi / 180
    # camera_height = 1.7
    mask_version = 1 # 0, 1

    crop_size = 480 # 480
    base_size = 520 # 520
    lang = "door,chair,ground,ceiling,other"
    labels = lang.split(",")
    vis = False

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # self.clip_version = "ViT-L/14"
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels)
    lang_token = lang_token.to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    # img_save_dirs = os.listdir(data_dir)
    # img_save_dirs = [os.path.join(data_dir, x) for x in img_save_dirs]
    print("Creating LSegEncNet model...")
    st = time.time()
    model = LSegEncNet(lang, arch_option=0,
                        block_depth=0,
                        activation='lrelu',
                        crop_size=crop_size)
    et = time.time()
    print(f"Creating time {et - st}s.")
    model_state_dict = model.state_dict()
    print("Loading pretrained model...")
    st = time.time()
    pretrained_state_dict = torch.load("lseg/checkpoints/demo_e200.ckpt")
    et = time.time()
    print(f"Loading time {et - st}s.")
    print("Filtering pretrained model...")
    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    print("Assigning pretrained model parameters to model...")
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.cuda()

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            # transforms.Resize([360,480]),
        ]
    )
    # print("All scene folders are: ", "\n".join(img_save_dirs))

    print(f"loading scene {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    pose_dir = os.path.join(img_save_dir, "pose")
    semantic_dir = os.path.join(img_save_dir, "semantic")
    obj2cls_path = os.path.join(img_save_dir, "obj2cls_dict.txt")
    # lseg_feat_dir = os.path.join(img_save_dir, "pixfeat")

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    semantic_list = sorted(os.listdir(semantic_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    # lseg_feat_list = sorted(os.listdir(lseg_feat_dir), key=lambda x: int(x.split('.')[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]
    semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]
    # lseg_feat_list = [os.path.join(lseg_feat_dir, x) for x in lseg_feat_list]


    map_save_dir = os.path.join(img_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    obj2cls = load_obj2cls_dict(obj2cls_path)
    id2cls = get_id2cls(obj2cls) # map from cls id to cls name

    # initialize a grid with zero at the center
    # cs = 0.05
    # gs = 1000
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    gt = np.zeros((gs, gs), dtype=np.int32)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)


    print("empty map saved")
    # compute the vfov of the camera
    bgr = cv2.imread(rgb_list[0])
    vfov_deg = get_vfov(hfov_deg, bgr.shape[0], bgr.shape[1])
    vfov_rad = vfov_deg * np.pi / 180.
    print("first image read")


    tf_list = []
    # load all images and depths and poses
    for iter_i, (rgb_path, depth_path, semantic_path, pose_path) in enumerate(zip(rgb_list, depth_list, semantic_list, pose_list)):
        print(f"Running iteration {iter_i}")
        st = time.time()
        # if iter_i > 100:
        #     break
        
        tic = Tic()
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


        # read pose
        pos, rot = load_pose(pose_path)  # z backward, y upward, x to the right
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot @ rot_ro_cam
        pos[1] += camera_height


        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0]) 

        tf = init_tf_inv @ pose
        theta = np.arctan2(tf[0, 2], tf[2, 2])

        # read depth
        depth = load_depth(depth_path)
        depth = depth / 10.
        vis_depth = depth[:, :, None]
        #cv2.imshow("depth", depth)
        #cv2.imshow("rgb", bgr)
        #k = cv2.waitKey(1)
        depth = depth * 10.
        tic.print_time("reading image, pose, and depth")

        # read semantic
        semantic = load_semantic(semantic_path)
        semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        tic.tic()

        pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size, norm_mean, norm_std, vis)
        # pix_feats = load_lseg_feat(feat_path)
        tic.print_time("generating lseg features")
        # tic.tic()
        
        # depth = cv2.resize(depth, (pix_feats.shape[3], pix_feats.shape[2]))
        
        # tic.print_time("resizing features")

        hf_2 = hfov_deg * np.pi / 360.
        vf_2 = vfov_rad / 2.

        # get the 4 points of the frustum on a top-down-map
        dmin = np.min([camera_height / np.tan(vf_2), np.min(depth[depth > 0])])
        # dmax = np.mean(depth[depth > 0])
        dmax = np.max(depth)

        # determine the ground region visible in the view (height of camera 1.7m)
        pts = get_frustum_4pts(dmin, dmax, theta, hf_2, vf_2)
        robot_pos = np.array([tf[0, 3], tf[2, 3]], dtype=np.float64)
        pts += robot_pos

        tmp = [pos2grid_id(gs, cs, x, y) for x, y in pts]
        pts_grid = np.array(tmp, dtype=np.int32)


        # project all point cloud onto the ground, once there are points in a cell,
        sample_rate = 100
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1]) 
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        # pc = pc[:, ::sample_rate]
        pc_global = transform_pc(pc, tf)

        # tic.tic()
        # feat_flat = np.reshape(pix_feats, (clip_feat_dim, -1)).T
        # feat_flat = feat_flat[shuffle_mask, :]
        # feat_flat = feat_flat[mask, :]
        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])
        # tic.print_time("sample feat")
        
        # mask v1
        if mask_version == 0:
            mask = generate_mask(gs, cs, hfov_rad, -theta+np.pi/2, depth, robot_pos[0], robot_pos[1])

        # mask v2
        elif mask_version == 1:
            mask = np.zeros_like(obstacles).astype(np.uint8)

        tic.tic()
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            if i % 1 != 0:
                continue
            x, y = pos2grid_id(gs, cs, p[0], p[2])
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            semantic_v = semantic[rgb_py, rgb_px]
            if semantic_v == 40:
                semantic_v = -1
            
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
                gt[y, x] = semantic_v


            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                # grid[y, x] = (grid[y, x] * weight[y, x] + feat_flat[i]) / (weight[y, x] + 1)
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            if p_local[1] > camera_height:
                continue
            # if i % 1 != 0 or p[1] > camera_height or p[1] < -0.5:
            #     continue
            if mask_version == 1:
                mask[y, x] = 255
            obstacles[y, x] = 0
        tic.print_time("projecting features")

        masked_obstacles = obstacles.copy() * 255
        # cv2.fillPoly(masked_obstacles, [pts_grid], color=120)
        cv2.polylines(masked_obstacles, [pts_grid], True, color=120)
        masked_obstacles[mask > 120] = 120

        #cv2.imshow("obstacles", masked_obstacles)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('v'):

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pc[0, ::100], pc[1, ::100], pc[2, ::100])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
        
        et = time.time()
        print(f"Iteration time: {et-st}")

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)


# img_save_dir = "/home/huang/hcg/projects/vln/data/clip_mapping/sim/5LpN3gDmAk7"
# img_save_dir = "/home/huang/hcg/projects/vln/data/clip_mapping/sim/JmbYfDe2QKZ"



def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5], vis=False):
    vis_image = image.copy()
    image = transform(image).unsqueeze(0).cuda()
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5
    
    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height


    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            # outputs = model(pad_img)
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
            count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    # output = model(pad_crop_img)
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        logits_outputs = logits_outputs[:,:,:height,:width]
    # outputs = resize_image(outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
    # outputs = resize_image(outputs, image.shape[0], image.shape[1], **{'mode': 'bilinear', 'align_corners': True})
    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]
    if vis:
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")
        cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
        cv2.waitKey()
        fig = plt.figure()
        plt.imshow(seg)
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 20})
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The directory of the rgb, depth, and pose")
    parser.add_argument("--camera_height", type=float, default=1.5, help="The height of the camera in the simulator")
    parser.add_argument("--depth_sample_rate", type=int, default=100, help="The height of the camera in the simulator")
    args = parser.parse_args()
    create_lseg_map_batch(args.data_dir, args.camera_height, depth_sample_rate=args.depth_sample_rate)



