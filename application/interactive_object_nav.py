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


def show_obs(robot, label: str = ""):
    """Display the robot's current RGB view in an OpenCV window."""
    rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(bgr, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Robot view", bgr)
    cv2.waitKey(1)


def find_best_start_pose(robot):
    """
    Scan ~30 evenly-spaced trajectory poses and return the one whose 2-D map
    cell is deepest inside free space (farthest from any obstacle).
    This avoids starting on top of furniture, which would make every
    move_to_object call return 0 actions (already at goal).
    """
    obs_map = robot.map.obstacles_map          # True = free, False = obstacle
    dist_map = distance_transform_edt(obs_map) # each free cell → distance to nearest obstacle

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

    # Pick the trajectory pose whose 2-D map cell is deepest inside free space.
    print("\nSearching for a good starting position...")
    start_tf = find_best_start_pose(robot)
    robot.set_agent_state(start_tf)
    show_obs(robot, "Ready")
    print("Scene:", robot.vlmaps_data_save_dirs[config.scene_id].name)

    # ── Instruction loop ─────────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 50)
        instruction = input("Enter navigation instruction (or 'quit'): ").strip()

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        # Parse instruction → list of object categories via GPT-4o-mini.
        print("Parsing instruction...")
        try:
            categories = parse_object_goal_instruction(instruction)
        except Exception as e:
            print(f"LLM error: {e}")
            continue

        print(f"Targets: {categories}")

        # Reset to start pose before each run so results are comparable.
        robot.set_agent_state(start_tf)
        robot.empty_recorded_actions()
        show_obs(robot, "Start")
        cv2.waitKey(500)

        # Navigate to each object in sequence.
        for cat in categories:
            cat = cat.strip()
            if not cat:
                continue
            print(f"\nPlanning path to: {cat}")

            # 1. Compute path silently (robot moves to destination internally).
            robot.empty_recorded_actions()
            robot.move_to_object(cat)
            planned_actions = robot.get_recorded_actions() or []
            n_actions = len(planned_actions)
            print(f"  Path computed: {n_actions} actions. Replaying...")
            if n_actions == 0:
                print(f"  [warn] No path found for '{cat}' — robot may already be "
                      f"at the target or the category is not in the map. Skipping.")
                continue

            # 2. Reset to start of this sub-goal and replay step by step.
            robot.set_agent_state(start_tf)
            for i, action in enumerate(planned_actions):
                if action == "stop":
                    continue
                robot.sim.step(action)
                show_obs(robot, f"[{i+1}/{n_actions}] -> {cat}")
                cv2.waitKey(80)  # ~12 fps — increase to slow down, decrease to speed up

            show_obs(robot, f"Arrived: {cat}")
            print(f"  Done.")
            cv2.waitKey(800)

            # Update start_tf to current position for the next sub-goal.
            from vlmaps.utils.habitat_utils import agent_state2tf
            agent_state = robot.sim.get_agent(0).get_state()
            start_tf = agent_state2tf(agent_state)

        print("\nInstruction complete. Press any key in the window to continue.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
