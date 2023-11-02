import os
from pathlib import Path
from subprocess import Popen
import hydra
from omegaconf import DictConfig

import numpy as np
import cv2
from PIL.Image import Image
from PIL.Image import fromarray

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import matplotlib

from matplotlib.widgets import TextBox
from mpl_point_clicker import clicker
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig

import magnum as mn
import habitat_sim
from habitat_sim import AgentState

from vlmaps.utils.habitat_utils import tf2agent_state


from vlmaps.map.vlmap import VLMap

# from habitat.utils.visualizations import maps

# from utils.utils import (
#     tf2agent_state,
#     make_cfg,
#     display_map,
#     display_sample,
#     save_obs,
#     save_state,
#     get_position_floor_objects,
# )
# from utils.clip_mapping_utils import (
#     load_map,
#     load_pose,
#     get_new_mask_pallete,
#     get_new_pallete,
#     grid_id2pos,
#     pos2grid_id,
# )
# from utils.controller.discrete_nav_controller import DiscreteNavController
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from typing import Tuple, List


class InteractiveMap:
    """
    Display the ground truth semantic top-down map of the scene. User can click on the map
    to select two points where the first denotes the starting position of the robot and
    the direction from the first to the second point denotes the robot orientation.
    """

    def __init__(
        self,
        data_dir: str,
        map_config: DictConfig,
        # obs_map_path: str,
        # gt_map_path: str,
        # pose_dir: str,
        # gs: int = 1000,
        # cs: float = 0.05,
        # camera_height: float = 1.5,
        load_gt_map: bool = False,
    ):
        matplotlib.use("TkAgg")
        # matplotlib.use("Qt5Agg")
        # load vlmaps_dataloader
        self.vlmaps_dataloader = VLMapsDataloaderHabitat(data_dir, map_config, load_gt_map=load_gt_map)
        self.gt_map = None
        if load_gt_map:
            self.gt_map = self.vlmaps_dataloader.gt

        # self.lang = mp3dcat

        # # set a controller
        # self.controller_config = {
        #     "gs": self.vlmaps_dataloader.gs,
        #     "cs": self.vlmaps_dataloader.cs,
        #     "forward_dist": 0.1,
        #     "turn_angle": 1,
        #     "goal_dist_thres": 0.05,
        # }
        # self.controller_config = DictConfig(self.controller_config)
        # self.controller = DiscreteNavController(self.controller_config)

    def _onclick(self, event):
        global icol, irow
        if event.button == 1:
            icol, irow = event.xdata, event.ydata
            print(f"[{icol}, {irow}],")
            self.coords.append((icol, irow))
            circle = Circle(
                (event.xdata, event.ydata),
                1.0 / self.vlmaps_dataloader.cs,
                alpha=0.5,
                facecolor=None,
                edgecolor="black",
            )
            self.ax.add_patch(circle)
            self.circles_list.append(circle)
        elif event.button == 2:
            st_id = len(self.coords)
            if st_id > self.start_ids[-1]:
                print(f"Start selecting the next group of points: {st_id}")
                self.start_ids.append(st_id)

        elif event.button == 3:
            coord = self.coords.pop()
            circle = self.circles_list.pop()
            circle.remove()
            print(f"remove point {coord}")

        return self.coords

    def collect_map_positions(
        self, showed_image: np.ndarray = None, patches: List[mpatches.Patch] = None
    ) -> List[Tuple[float, float]]:
        """
        Visualize a top down map and let user select a series of points on the map
        Return a list of point coordinates (col, row) in the map.
        """
        # setup matplotlib figure for interactive actions
        self.fig = plt.figure()
        self.ax = plt.gca()
        self.circles_list = []
        self.coords = []
        self.start_ids = [0]
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        if showed_image is None:
            mask_gt, patches_gt = self.get_labeled_gt_map()
            seg_gt = mask_gt.convert("RGBA")
            seg_gt_np = np.array(seg_gt)

            klicker = clicker(self.ax, ["event"], markers=["o"], colors=["r"])
            self.ax.legend(
                handles=patches_gt,
                loc="upper left",
                bbox_to_anchor=(1.0, 1),
                prop={"size": 10},
            )
            self.ax.axis("off")
            self.ax.imshow(seg_gt)
        elif patches is not None:
            print(showed_image.shape)
            showed_image = fromarray(showed_image[:, :, :3])
            seg_gt = showed_image.convert("RGBA")

            klicker = clicker(self.ax, ["event"], markers=["o"], colors=["r"])
            self.ax.legend(
                handles=patches,
                loc="upper left",
                bbox_to_anchor=(1.0, 1),
                prop={"size": 10},
            )
            self.ax.axis("off")
            self.ax.imshow(showed_image)

        else:
            self.ax.axis("off")
            self.ax.imshow(showed_image)

        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        backend = matplotlib.get_backend()
        x, y = 1500, 0
        if backend == "TkAgg":
            mng.window.wm_geometry("+%d+%d" % (x, y))
            pass
        elif backend == "WXAgg":
            mng.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            mng.window.move(x, y)

        mng.resize(1500, 2000)
        print("starting to show figure")
        print("(x, y)")
        plt.show()
        return self.coords

    def _submit(self, expression: str):
        self.instruction = expression
        print(self.instruction)

    def collect_goals_and_instructions(
        self,
        showed_image: np.ndarray = None,
        patches: List[mpatches.Patch] = None,
        zoom_times: float = 1,
    ) -> Tuple[np.ndarray, List[List[List[float]]], str]:
        self.instruction = ""
        fig_2 = plt.figure()
        axbox = fig_2.add_axes([0.1, 0.05, 0.8, 0.075])
        text_box = TextBox(axbox, "Instruction")
        text_box.on_submit(self._submit)
        text_box.set_val("")  # Trigger `submit` with the initial string.
        mng = plt.get_current_fig_manager()
        mng.resize(3000, 240)
        self.collect_map_positions(showed_image, patches)
        start_end_ids = zip(self.start_ids, [*self.start_ids[1:], len(self.coords)])
        xmin = self.vlmaps_dataloader.xmin
        ymin = self.vlmaps_dataloader.ymin
        all_group_coords_full = []
        for group_i, (st, ed) in enumerate(start_end_ids):
            print(st, ed)
            group_coords = self.coords[st:ed]  # [(col, row), ...]
            if group_i == 0:
                src = [group_coords[0][0] / zoom_times, group_coords[0][1] / zoom_times]
                tar = [group_coords[1][0] / zoom_times, group_coords[1][1] / zoom_times]
                init_tf_hab, agent_state = self.get_habitat_robot_state(src, tar)
                print("init cropped pos: ")
                print(src, tar)
                continue
            group_coords_full = [[x[1] / zoom_times + xmin, x[0] / zoom_times + ymin] for x in group_coords]
            all_group_coords_full.append(group_coords_full)
        return init_tf_hab, all_group_coords_full, self.instruction

    def get_labeled_gt_map(self) -> Tuple[Image, List[mpatches.Patch]]:
        new_pallete = get_new_pallete(len(self.lang))
        mask_gt, patches_gt = get_new_mask_pallete(
            self.vlmaps_dataloader.gt_cropped,
            new_pallete,
            out_label_flag=True,
            labels=self.lang,
        )
        return mask_gt, patches_gt

    # def get_habitat_robot_position(self, coord: Tuple[float, float]) -> np.ndarray:
    #     row_full = self.xmin + coord[1]
    #     col_full = self.ymin + coord[0]
    #     x, z = grid_id2pos(self.gs, self.cs, col_full, row_full)
    #     tf = np.eye(4)
    #     tf[:3, 3] = [x, 0, z]

    #     tf_hab = (
    #         self.map_init_pose @ self.tf_ro_cam @ tf @ np.linalg.inv(self.tf_ro_cam)
    #     )
    #     return tf_hab[:3, 3].flatten()

    def get_habitat_robot_state(
        self, src_coord: Tuple[float, float], tar_coord: Tuple[float, float]
    ) -> Tuple[np.ndarray, habitat_sim.AgentState]:
        """
        Get the robot state obtained from user interation in Habitat sim format
        Return (4x4 numpy array denoting transformation, robot_state in habitat format)
        """
        theta = np.arctan2(-tar_coord[0] + src_coord[0], -tar_coord[1] + src_coord[1])
        print("theta: ", theta)
        print("theta deg: ", np.rad2deg(theta))
        theta_deg = np.rad2deg(theta)
        self.vlmaps_dataloader.from_cropped_map_pose(src_coord[1], src_coord[0], theta_deg)
        tf_hab = self.vlmaps_dataloader.to_habitat_tf()
        agent_state = tf2agent_state(tf_hab)
        return tf_hab, agent_state

    def get_habitat_robot_position_batch(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        positions = []
        for coord_i, coord in enumerate(coords):
            pose = self.vlmaps_dataloader.convert_cropped_map_pose_to_habitat_pose(coord[1], coord[0], 0)
            position = pose[:3, 3].flatten()
            positions.append(position)

        positions = np.array(positions).reshape((-1, 3))
        return positions

    # def convert_habitat_pose_to_full_map_pose(
    #     self, tf_hab: np.ndarray
    # ) -> Tuple[float, float, float]:
    #     tf_cam = (
    #         np.linalg.inv(self.map_init_pose @ self.tf_ro_cam) @ tf_hab @ self.tf_ro_cam
    #     )
    #     x, z = tf_cam[[0, 2], 3]
    #     theta = np.arctan2(tf_cam[0, 2], tf_cam[2, 2])
    #     theta_deg = np.rad2deg(theta)
    #     col, row = pos2grid_id(self.gs, self.cs, x, z)
    #     return (row, col, theta_deg)

    def convert_habitat_position_to_full_map_position(self, p: np.ndarray) -> Tuple[float, float]:
        tf_hab = np.eye(4)
        tf_hab[:3, 3] = p.flatten()
        (
            row,
            col,
            angle_deg,
        ) = self.vlmaps_dataloader.convert_habitat_tf_to_full_map_pose(tf_hab)
        return row, col

    def load_habitat_scene_mesh(self, scene_dir: str, scene_name: str, sim_setting: dict = None):
        self.scene_name = scene_name
        self.test_scene = os.path.join(scene_dir, scene_name, scene_name + ".glb")
        self.test_navmesh = os.path.join(scene_dir, scene_name, scene_name + ".navmesh")
        self.sim_setting = sim_setting
        if not sim_setting:
            self.sim_setting = {
                "scene": self.test_scene,
                "default_agent": 0,
                "sensor_height": 1.5,
                "color_sensor": True,
                "depth_sensor": False,
                "semantic_sensor": False,
                "lidar_sensor": False,
                "move_forward": 0.1,
                "turn_left": 1,
                "turn_right": 1,
                "width": 640,
                "height": 480,
                "enable_physics": False,
                "seed": 1,
                "lidar_fov": 360,
                "depth_img_for_lidar_n": 20,
                "img_save_dir": "/tmp",
            }

        cfg = make_cfg(self.sim_setting)
        self.sim = habitat_sim.Simulator(cfg)
        agent = self.sim.initialize_agent(self.sim_setting["default_agent"])
        self.sim.pathfinder.load_nav_mesh(self.test_navmesh)

    def set_agent_state(self, agent_state: habitat_sim.AgentState):
        self.sim.get_agent(0).set_state(agent_state)

    def get_obs(self):
        obs = self.sim.get_sensor_observations(0)
        rgb = obs["color_sensor"]
        return rgb

    def get_object_bboxes(self, categories: List[str]):
        same_floor_objects_list = get_position_floor_objects(
            self.sim.semantic_scene,
            self.vlmaps_dataloader.map_init_pose[:3, 3],
            self.vlmaps_dataloader.camera_height + 0.5,
        )
        object_bboxes_hab = []
        for cat_i, cat in enumerate(categories):
            cat_objects = [x for x in same_floor_objects_list if x.category.name() == cat]
            cat_centers_hab = [object.aabb.center.tolist() for object in cat_objects]
            cat_sizes = [object.aabb.sizes.tolist() for object in cat_objects]
            bboxes = {}
            bboxes["centers"] = cat_centers_hab
            bboxes["sizes"] = cat_sizes
            object_bboxes_hab.append(bboxes)
        return object_bboxes_hab

    def get_floor_object_bboxes(self, categories: List[str] = [], ignore_categories: List[str] = []):
        same_floor_objects_list = get_position_floor_objects(
            self.sim.semantic_scene,
            self.vlmaps_dataloader.map_init_pose[:3, 3],
            self.vlmaps_dataloader.camera_height + 0.5,
        )
        bbox_centers = [x.aabb.center for x in same_floor_objects_list]
        bbox_sizes = [x.aabb.sizes for x in same_floor_objects_list]
        bbox_categories = [x.category.name() for x in same_floor_objects_list]
        if len(categories) > 0:
            bbox_centers = [x for x, y in zip(bbox_centers, bbox_categories) if y in categories]
            bbox_sizes = [x for x, y in zip(bbox_sizes, bbox_categories) if y in categories]
            bbox_categories = [x for x in bbox_categories if x in categories]
        if len(ignore_categories) > 0:
            bbox_centers = [x for x, y in zip(bbox_centers, bbox_categories) if y not in ignore_categories]
            bbox_sizes = [x for x, y in zip(bbox_sizes, bbox_categories) if y not in ignore_categories]
            bbox_categories = [x for x in bbox_categories if x not in ignore_categories]

        object_bboxes_hab = {}
        object_bboxes_hab["centers"] = bbox_centers
        object_bboxes_hab["sizes"] = bbox_sizes
        object_bboxes_hab["categories"] = bbox_categories

        return object_bboxes_hab

    def get_floor_region_bboxes(self):
        dataloader = self.vlmaps_dataloader
        scene = self.sim.semantic_scene
        same_floor_regions_list = get_position_floor_objects(
            scene, dataloader.map_init_pose[:3, 3], dataloader.camera_height + 0.5, concept_type="region"
        )
        bbox_centers = [x.aabb.center for x in same_floor_regions_list]
        bbox_sizes = [x.aabb.sizes for x in same_floor_regions_list]
        bbox_categories = [x.category.name() for x in same_floor_regions_list]

        region_bboxes_hab = {}
        region_bboxes_hab["centers"] = bbox_centers
        region_bboxes_hab["sizes"] = bbox_sizes
        region_bboxes_hab["categories"] = bbox_categories
        return region_bboxes_hab

    def convert_bboxes_hab_to_full_map_bboxes(
        self, bbox_centers_hab: List[np.ndarray], bbox_sizes_hab: List[np.ndarray]
    ) -> List[np.ndarray]:
        bboxes = []
        for bbox_i, (bbox_center_hab, bbox_size_hab) in enumerate(zip(bbox_centers_hab, bbox_sizes_hab)):
            top_left_x = bbox_center_hab[0] - bbox_size_hab[0] / 2.0
            top_left_z = bbox_center_hab[2] - bbox_size_hab[2] / 2.0
            bottom_right_x = bbox_center_hab[0] + bbox_size_hab[0] / 2.0
            bottom_right_z = bbox_center_hab[2] + bbox_size_hab[2] / 2.0
            top_left_row, top_left_col = self.vlmaps_dataloader.convert_habitat_pos_list_to_full_map_pos_list(
                [np.array([top_left_x, 0, top_left_z])]
            )[0]
            bottom_right_row, bottom_right_col = self.vlmaps_dataloader.convert_habitat_pos_list_to_full_map_pos_list(
                [np.array([bottom_right_x, 0, bottom_right_z])]
            )[0]
            bboxes.append(np.array([top_left_row, top_left_col, bottom_right_row, bottom_right_col]))
        return bboxes

    def convert_bboxes_hab_to_cropped_map_bboxes(
        self, bbox_centers_hab: List[np.ndarray], bbox_sizes_hab: List[np.ndarray]
    ) -> List[np.ndarray]:
        bboxes = self.convert_bboxes_hab_to_full_map_bboxes(bbox_centers_hab, bbox_sizes_hab)
        bboxes_cropped = []
        for bbox in bboxes:
            top_left = self.vlmaps_dataloader.convert_full_map_pos_to_cropped_map_pos(bbox[:2])
            bottom_right = self.vlmaps_dataloader.convert_full_map_pos_to_cropped_map_pos(bbox[2:])
            bboxes_cropped.append(np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]]))
        return bboxes_cropped

    def display_obs(self):
        rgb = self.get_obs()
        display_sample(self.sim_setting, rgb)

    def get_path(self, positions: np.ndarray) -> List[np.ndarray]:
        tf_hab, agent_state = self.get_habitat_robot_state(positions[0], positions[1])
        self.sim.get_agent(0).set_state(agent_state)
        full_path_points = []
        for state_i, position in enumerate(positions):
            if state_i == positions.shape[0] - 1:
                full_path_points.append(position)
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start = position
            path.requested_end = positions[state_i + 1]
            found = self.sim.pathfinder.find_path(path)
            if found:
                full_path_points.extend(path.points[:-1])

        return full_path_points

    def get_actions_v2(self, positions: List[np.ndarray]) -> List[str]:
        agent_state = self.sim.get_agent(0).get_state()
        init_hab_tf = agent_state2tf(agent_state)
        start_pose_on_map = self.vlmaps_dataloader.convert_habitat_tf_to_full_map_pose(init_hab_tf)
        paths = [self.convert_habitat_position_to_full_map_position(x) for x in positions]

        actions, poses = self.controller.convert_paths_to_actions(start_pose_on_map, paths)
        return actions

    def get_actions(self, positions: List[np.ndarray]) -> List[str]:
        # tf_hab, agent_state = self.get_habitat_robot_state(positions[0], positions[1])
        agent = self.sim.get_agent(0)
        # agent.set_state(agent_state)

        follower = habitat_sim.nav.GreedyGeodesicFollower(self.sim.pathfinder, agent, fix_thrashing=False)
        full_actions_list = []
        for position_i, position in enumerate(positions):
            if position_i == 0:
                continue

            nearby_goal = self.sim.pathfinder.snap_point(position)
            print(f"trying to find path to position {position_i}")
            print(f"goal position before snap: ", position)
            print(f"goal position after snap: ", nearby_goal)
            print(f"current robot position: ", agent.get_state().position)
            print(f"current robot rotation: ", agent.get_state().rotation)

            # try:
            #     actions_list = follower.find_path(nearby_goal)
            # except habitat_sim.errors.GreedyFollowerError:

            while True:
                try:
                    next_action = follower.next_action_along(nearby_goal)
                except habitat_sim.errors.GreedyFollowerError:
                    print("greedy follower error")
                    break
                if next_action is None:
                    break
                obs = self.sim.step(next_action)
                display_sample(self.sim_setting, obs["color_sensor"], waitkey=False)
                full_actions_list.append(next_action)
            # for action_i, action in enumerate(actions_list):
            #     if action is None:
            #         break
            #     self.sim.step(action)
            #     full_actions_list.append(action)

        return full_actions_list

    def play_actions(
        self,
        start_agent_state: habitat_sim.AgentState,
        actions_list: List[str],
        waitkey: bool = False,
    ) -> List[np.ndarray]:
        self.set_agent_state(start_agent_state)
        obss = []
        poses = []
        for action_i, action in enumerate(actions_list):
            if action is None:
                break
            obs = self.sim.step(action)
            display_sample(self.sim_setting, obs["color_sensor"], waitkey=waitkey)
            agent_state = self.sim.get_agent(0).get_state()
            obss.append(obs)
            poses.append(agent_state)
        return obss, poses

    def save_actions_obs(
        self,
        save_dir: str,
        start_agent_state: habitat_sim.AgentState,
        actions_list: List[str],
    ):
        self.set_agent_state(start_agent_state)
        for action_i, action in enumerate(actions_list):
            if action is None:
                break
            obs = self.sim.step(action)
            agent_state = self.sim.get_agent(0).get_state()

            print(f"saving {self.scene_name} observation {action_i:06}")
            save_obs(save_dir, self.sim_setting, obs, action_i)
            save_state(save_dir, self.sim_setting, agent_state, action_i)

    def save_obs_pose(self, save_dir: str, obss: List[dict], poses: List[habitat_sim.AgentState]):
        for action_i, (obs, agent_state) in enumerate(zip(obss, poses)):
            print(f"saving {self.scene_name} observation {action_i:06}")
            save_obs(save_dir, self.sim_setting, obs, action_i)
            save_state(save_dir, self.sim_setting, agent_state, action_i)

    def draw_path(self, path_points: List[np.ndarray]):
        height = self.sim.get_agent(0).get_state().position[1]
        hablab_topdown_map = maps.get_topdown_map(self.sim.pathfinder, height, meters_per_pixel=0.05)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        top_down_map = recolor_map[hablab_topdown_map]

        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=self.sim.pathfinder,
            )
            for path_point in path_points
        ]
        grid_tangent = mn.Vector2(trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0])

        path_initial_tangent = grid_tangent / grid_tangent.length()
        initial_angle = np.arctan2(path_initial_tangent[0], path_initial_tangent[1])

        maps.draw_path(top_down_map, trajectory)
        agent_state = self.sim.get_agent(0).get_state()
        grid_pos = maps.to_grid(
            agent_state.position[2],
            agent_state.position[0],
            grid_dimensions,
            pathfinder=self.sim.pathfinder,
        )
        quat = agent_state.rotation
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])

        angle = np.pi + 2 * np.arcsin(quat.y)
        maps.draw_agent(top_down_map, grid_pos, angle, agent_radius_px=8)

        display_map(top_down_map)

    def visualize_floor_object_bboxes(
        self, zoom: float = 1.0, ignore: List[str] = ["void", "wall", "floor", "misc", "appliances", "objects"]
    ):
        dataloader = self.vlmaps_dataloader
        object_bboxes_hab = self.get_floor_object_bboxes()
        object_bboxes_cropped = self.convert_bboxes_hab_to_cropped_map_bboxes(
            object_bboxes_hab["centers"], object_bboxes_hab["sizes"]
        )
        obstacles = dataloader.get_obstacles_cropped_no_floor()
        obstacles = cv2.cvtColor((obstacles * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        obstacles = cv2.resize(obstacles, (0, 0), fx=zoom, fy=zoom)
        for bbox_i, (bbox, cat) in enumerate(zip(object_bboxes_cropped, object_bboxes_hab["categories"])):
            if cat in ignore:
                continue
            obstacles = cv2.rectangle(
                obstacles,
                (int(bbox[1] * zoom), int(bbox[0] * zoom)),
                (int(bbox[3] * zoom), int(bbox[2] * zoom)),
                (0, 0, 255),
                2,
            )
            obstacles = cv2.putText(
                obstacles,
                cat,
                (int(bbox[1] * zoom + 10), int(bbox[0] * zoom + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imshow("obstacles", obstacles)
        cv2.waitKey()

    def visualize_floor_region_bboxes(self, zoom: float = 1.0):
        dataloader = self.vlmaps_dataloader
        region_bboxes_hab = self.get_floor_region_bboxes()
        region_bboxes_cropped = self.convert_bboxes_hab_to_cropped_map_bboxes(
            region_bboxes_hab["centers"], region_bboxes_hab["sizes"]
        )
        obstacles = dataloader.get_obstacles_cropped_no_floor()
        obstacles = cv2.cvtColor((obstacles * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        obstacles = cv2.resize(obstacles, (0, 0), fx=zoom, fy=zoom)
        for bbox_i, (bbox, cat) in enumerate(zip(region_bboxes_cropped, region_bboxes_hab["categories"])):
            obstacles = cv2.rectangle(
                obstacles,
                (int(bbox[1] * zoom), int(bbox[0] * zoom)),
                (int(bbox[3] * zoom), int(bbox[2] * zoom)),
                (0, 0, 255),
                2,
            )
            obstacles = cv2.putText(
                obstacles,
                cat,
                (int(bbox[1] * zoom + 10), int(bbox[0] * zoom + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imshow("obstacles", obstacles)
        cv2.waitKey()


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    interactive_map = InteractiveMap(data_dirs[config.scene_id], config.map_config)
    # obstacle_cropped = interactive_map.vlmaps_dataloader.get_obstacles_cropped()
    # bgr = interactive_map.vlmaps_dataloader.get_color_topdown_bgr_cropped()
    color_top_down = interactive_map.vlmaps_dataloader.map.generate_rgb_topdown_map() / 255.0
    color_top_down = color_top_down[
        interactive_map.vlmaps_dataloader.rmin : interactive_map.vlmaps_dataloader.xmax + 1,
        interactive_map.vlmaps_dataloader.cmin : interactive_map.vlmaps_dataloader.ymax + 1,
    ]

    interactive_map.collect_map_positions(color_top_down)
    tf_hab, agent_state = interactive_map.get_habitat_robot_state(interactive_map.coords[0], interactive_map.coords[1])
    print("tf_hab: ", tf_hab)


if __name__ == "__main__":
    main()
