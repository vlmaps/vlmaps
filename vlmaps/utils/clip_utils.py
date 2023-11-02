import os
from argparse import ArgumentParser
import numpy as np
import cv2
from PIL import Image
import torch

import clip

multiple_templates = [
    "There is {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "I took a picture of of {}.",
    "I took a picture of of my {}.",  # itap: I took a picture of
    "I took a picture of of the {}.",
    "a photo of {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {}.",
    "a good photo of the {}.",
    "a bad photo of {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {}.",
    "a bright photo of the {}.",
    "a dark photo of {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def match_text_to_imgs(language_instr, images_list):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    imgs_feats = get_imgs_feats(images_list)
    text_feats = get_text_feats([language_instr])
    scores = imgs_feats @ text_feats.T
    scores = scores.squeeze()
    return scores, imgs_feats, text_feats


def get_nn_img(raw_imgs, text_feats, img_feats):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    scores = img_feats @ text_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_imgs = [raw_imgs[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_ids, high_to_low_imgs, high_to_low_scores


def get_img_feats(img, preprocess, clip_model):
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_imgs_feats(raw_imgs, preprocess, clip_model, clip_feat_dim):
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    for img_id, img in enumerate(raw_imgs):
        imgs_feats[img_id, :] = get_img_feats(img, preprocess, clip_model)
    return imgs_feats


def get_imgs_feats_batch(raw_imgs, preprocess, clip_model, clip_feat_dim, batch_size=64):
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    img_batch = []
    for img_id, img in enumerate(raw_imgs):
        if img.shape[0] == 0 or img.shape[1] == 0:
            img = [[[0, 0, 0]]]
        img_pil = Image.fromarray(np.uint8(img))
        img_in = preprocess(img_pil)[None, ...]
        img_batch.append(img_in)
        if len(img_batch) == batch_size or img_id == len(raw_imgs) - 1:
            img_batch = torch.cat(img_batch, dim=0)
            with torch.no_grad():
                batch_feats = clip_model.encode_image(img_batch.cuda()).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            imgs_feats[img_id - len(img_batch) + 1 : img_id + 1, :] = batch_feats
            img_batch = []
    return imgs_feats


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    if torch.cuda.is_available():
        text_tokens = clip.tokenize(in_text).cuda()
    elif torch.backends.mps.is_available():
        text_tokens = clip.tokenize(in_text).to("mps")
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats


def get_text_feats_multiple_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    mul_tmp = multiple_templates.copy()
    multi_temp_landmarks_other = [x.format(lm) for lm in in_text for x in mul_tmp]
    text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)
    # average the features
    text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    return text_feats


def main():
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, help="the path to the image")
    parser.add_argument("--categories", type=str, help="categories separated by comma, e.g. table,chair,picture")
    args = parser.parse_args()
    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
    }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    # load image
    bgr = cv2.imread(args.img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_feats = get_img_feats(rgb, preprocess, clip_model)
    categories = args.categories.split(",")
    text_feats = get_text_feats_multiple_templates(categories, clip_model, clip_feat_dim)

    scores_mat = img_feats @ text_feats.T  # (1, categories_num)
    print(scores_mat)


def get_lseg_score(
    clip_model,
    landmarks: list,
    lseg_map: np.array,
    clip_feat_dim: int,
    use_multiple_templates: bool = False,
    avg_mode: int = 0,
    add_other=True,
):
    """
    Inputs:
        landmarks: a list of strings that describe the landmarks
        lseg_map: a numpy array with shape (h, w, clip_dim) or (N, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
    Return:
        scores_list: (h, w, class_num) storing the score for each location for each class
    """
    landmarks_other = landmarks
    if add_other and landmarks_other[-1] != "other":
        landmarks_other = landmarks + ["other"]

    if use_multiple_templates:
        mul_tmp = multiple_templates.copy()
        multi_temp_landmarks_other = [x.format(lm) for lm in landmarks_other for x in mul_tmp]
        text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)

        # average the features
        if avg_mode == 0:
            text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
            text_feats = np.mean(text_feats, axis=1)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

        # average the features
        if avg_mode == 1:
            scores_list = scores_list.reshape((-1, len(landmarks_other), len(mul_tmp)))
            scores_list = np.mean(scores_list, axis=2)
    else:
        text_feats = get_text_feats(landmarks_other, clip_model, clip_feat_dim)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

    return scores_list


if __name__ == "__main__":
    main()
