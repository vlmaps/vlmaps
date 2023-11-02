import os
import cv2
import numpy as np
import openai
from vlmaps.utils.clip_utils import get_text_feats, multiple_templates


def find_similar_category_id(class_name, classes_list):
    """
    Return the id of the most similar name to class_name in classes_list
    """
    if class_name in classes_list:
        return classes_list.index(class_name)
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    classes_list_str = ",".join(classes_list)
    question = f"""
    Q: What is television most relevant to among tv_monitor,plant,chair. A:tv_monitor\n
    Q: What is drawer most relevant to among tv_monitor,chest_of_drawers,chair. A:chest_of_drawers\n
    Q: What is {class_name} most relevant to among {classes_list_str}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop="\n",
    )
    result = response["choices"][0]["text"].strip()
    print(f"Similar category of {class_name} is {result}")
    return classes_list.index(result)


def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy


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
        lseg_map: a numpy array with shape (h, w, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
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


def segment_lseg_map(
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
        lseg_map: a numpy array with shape (h, w, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
    """
    scores_list = get_lseg_score(
        clip_model, landmarks, lseg_map, clip_feat_dim, use_multiple_templates, avg_mode, add_other
    )

    empty_mask = np.sum(lseg_map, axis=2) < 1e-6
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((lseg_map.shape[0], lseg_map.shape[1]))
    predicts[empty_mask] = -1

    return predicts


def get_dynamic_obstacles_map_3d(
    clip_model,
    obstacles_cropped,
    potential_obstacle_classes,
    obstacle_classes,
    grid_feat,
    grid_pos,
    rmin,
    cmin,
    clip_feat_dim,
    use_multiple_templates=True,
    avg_mode=0,
    vis=False,
):
    all_obstacles_mask = obstacles_cropped == 0
    scores_mat = get_lseg_score(
        clip_model,
        potential_obstacle_classes,
        grid_feat,
        clip_feat_dim,
        use_multiple_templates=use_multiple_templates,
        avg_mode=avg_mode,
    )
    predict = np.argmax(scores_mat, axis=1)
    obs_inds = []
    for obs_name in obstacle_classes:
        for i, po_obs_name in enumerate(potential_obstacle_classes):
            if obs_name == po_obs_name:
                obs_inds.append(i)

    print("obs_inds: ", obs_inds)
    pts_mask = np.zeros_like(predict, dtype=bool)
    for id in obs_inds:
        tmp = predict == id
        pts_mask = np.logical_or(pts_mask, tmp)

    # new_obstacles = obstacles_segment_map != 1
    new_obstacles = np.zeros_like(obstacles_cropped, dtype=bool)
    obs_pts = grid_pos[pts_mask]
    new_obstacles[obs_pts[:, 0] - rmin, obs_pts[:, 1] - cmin] = 1
    new_obstacles = np.logical_and(new_obstacles, all_obstacles_mask)
    new_obstacles = np.logical_not(new_obstacles)

    if vis:
        cv2.imshow("new obstacles_cropped", (new_obstacles * 255).astype(np.uint8))
        cv2.waitKey()
    return new_obstacles
