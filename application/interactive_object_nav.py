"""
interactive_object_nav.py
=========================
Type a natural-language instruction and watch the robot navigate in real time.

Usage (from the repository root)
---------------------------------
    python application/interactive_object_nav.py scene_id=1

Controls
--------
    - A new OpenCV window shows the robot's camera view after every action.
    - A second window shows the semantic top-down map with the target heatmap
      and the robot's current position.
    - Press any key in the window to advance to the next step.
    - Type a new instruction at the prompt to run another navigation.
    - Type 'quit' or 'exit' to stop.
"""

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from scipy.ndimage import distance_transform_edt

from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d, pool_3d_rgb_to_2d


# ── Visualization helpers ─────────────────────────────────────────────────────

def build_rgb_map_2d(robot) -> np.ndarray:
    """Build a top-down RGB map from the loaded VLMap (done once per scene)."""
    return pool_3d_rgb_to_2d(robot.map.grid_rgb, robot.map.grid_pos, robot.map.gs)


def show_obs(robot, label: str = ""):
    """Display the robot's current RGB view in an OpenCV window."""
    rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(bgr, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Robot view", bgr)
    cv2.waitKey(1)


def show_map(robot, rgb_map_2d: np.ndarray, category: str = "",
             path_cells: list = None, label: str = ""):
    """
    Display a top-down semantic map with:
      - RGB background of the scene
      - Heatmap overlay for the queried category (if provided)
      - Planned path drawn as a blue polyline
      - Robot's current position as a green circle
    """
    gs = robot.map.gs

    # ── Base: RGB top-down map ────────────────────────────────────────────────
    canvas = rgb_map_2d.astype(np.float32).copy()

    # ── Semantic heatmap overlay ──────────────────────────────────────────────
    if category:
        try:
            mask_3d = robot.map.index_map(category, with_init_cat=True)
            mask_2d = pool_3d_label_to_2d(mask_3d, robot.map.grid_pos, gs)
            # Smooth heatmap from binary mask
            from scipy.ndimage import distance_transform_edt as edt
            dist = edt(~mask_2d)
            heatmap = np.clip(1.0 - dist * 0.05, 0, 1).astype(np.float32)
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heat_rgb = heat_bgr[:, :, ::-1].astype(np.float32)
            canvas = canvas * 0.5 + heat_rgb * 0.5
        except Exception:
            pass  # skip overlay if indexing fails

    # ── Planned path ──────────────────────────────────────────────────────────
    if path_cells and len(path_cells) > 1:
        pts = np.array([[c[1], c[0]] for c in path_cells], dtype=np.int32)
        canvas_bgr = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.polylines(canvas_bgr, [pts], False, (255, 100, 0), 1)
        canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ── Robot position ────────────────────────────────────────────────────────
    row = int(robot.curr_pos_on_map[0])
    col = int(robot.curr_pos_on_map[1])
    canvas_bgr = cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
    cv2.circle(canvas_bgr, (col, row), 5, (0, 255, 0), -1)   # filled green dot
    cv2.circle(canvas_bgr, (col, row), 7, (255, 255, 255), 1) # white outline

    # ── Label ─────────────────────────────────────────────────────────────────
    if label:
        cv2.putText(canvas_bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas_bgr, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Scale up so the map is easier to read (grid is 500x500 by default)
    scale = max(1, 600 // gs)
    if scale > 1:
        canvas_bgr = cv2.resize(canvas_bgr, (gs * scale, gs * scale),
                                 interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Semantic Map", canvas_bgr)
    cv2.waitKey(1)


# ── Navigation helpers ────────────────────────────────────────────────────────

def find_best_start_pose(robot):
    """
    Scan ~30 evenly-spaced trajectory poses and return the one whose 2-D map
    cell is deepest inside free space (farthest from any obstacle).
    This avoids starting on top of furniture, which would make every
    move_to_object call return 0 actions (already at goal).
    """
    obs_map = robot.map.obstacles_map
    dist_map = distance_transform_edt(obs_map)

    poses = robot.vlmaps_dataloader.base_poses
    n = len(poses)
    step = max(1, n // 30)

    best_idx = 0
    best_dist = -1.0
    gs = obs_map.shape[0]

    for i in range(0, n, step):
        tf = cvt_pose_vec2tf(poses[i])
        robot.set_agent_state(tf)
        robot._set_nav_curr_pose()
        row = int(robot.curr_pos_on_map[0])
        col = int(robot.curr_pos_on_map[1])
        if 0 <= row < gs and 0 <= col < gs and obs_map[row, col]:
            d = float(dist_map[row, col])
            if d > best_dist:
                best_dist = d
                best_idx = i

    print(f"Best start: pose[{best_idx}/{n}]  map=({int(robot.curr_pos_on_map[0])},{int(robot.curr_pos_on_map[1])})  dist_to_obstacle={best_dist:.1f} cells")
    return cvt_pose_vec2tf(poses[best_idx])


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="object_goal_navigation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    robot.map.init_categories(mp3dcat.copy())

    print("\nBuilding top-down RGB map...")
    rgb_map_2d = build_rgb_map_2d(robot)

    print("\nSearching for a good starting position...")
    start_tf = find_best_start_pose(robot)
    robot.set_agent_state(start_tf)
    robot._set_nav_curr_pose()

    show_obs(robot, "Ready")
    show_map(robot, rgb_map_2d, label="Ready")
    print("Scene:", robot.vlmaps_data_save_dirs[config.scene_id].name)

    # ── Instruction loop ─────────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 50)
        instruction = input("Enter navigation instruction (or 'quit'): ").strip()

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        print("Parsing instruction...")
        try:
            categories = parse_object_goal_instruction(instruction)
        except Exception as e:
            print(f"LLM error: {e}")
            continue

        print(f"Targets: {categories}")

        robot.set_agent_state(start_tf)
        robot._set_nav_curr_pose()
        robot.empty_recorded_actions()
        show_obs(robot, "Start")
        show_map(robot, rgb_map_2d, label="Start")
        cv2.waitKey(500)

        for cat in categories:
            cat = cat.strip()
            if not cat:
                continue
            print(f"\nPlanning path to: {cat}")

            # Show heatmap for this category while planning
            show_map(robot, rgb_map_2d, category=cat, label=f"Planning: {cat}")
            cv2.waitKey(200)

            robot.empty_recorded_actions()
            robot.move_to_object(cat)
            planned_actions = robot.get_recorded_actions() or []
            n_actions = len(planned_actions)
            print(f"  Path computed: {n_actions} actions. Replaying...")

            if n_actions == 0:
                print(f"  [warn] No path found for '{cat}' — skipping.")
                continue

            robot.set_agent_state(start_tf)
            robot._set_nav_curr_pose()

            for i, action in enumerate(planned_actions):
                if action == "stop":
                    continue
                robot.sim.step(action)
                robot._set_nav_curr_pose()
                show_obs(robot, f"[{i+1}/{n_actions}] -> {cat}")
                show_map(robot, rgb_map_2d, category=cat,
                         label=f"[{i+1}/{n_actions}] -> {cat}")
                cv2.waitKey(80)

            show_obs(robot, f"Arrived: {cat}")
            show_map(robot, rgb_map_2d, category=cat, label=f"Arrived: {cat}")
            print(f"  Done.")
            cv2.waitKey(800)

            from vlmaps.utils.habitat_utils import agent_state2tf
            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)
            robot._set_nav_curr_pose()

        print("\nInstruction complete. Press any key in the window to continue.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
