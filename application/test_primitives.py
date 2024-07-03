from pathlib import Path
import cv2
import hydra
from omegaconf import DictConfig
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.map.interactive_map import InteractiveMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.mapping_utils import (
    cvt_pose_vec2tf,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config.scene_id)
    robot.map.init_categories(mp3dcat[1:-1])
    # hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[0])
    # robot.set_agent_state(hab_tf)
    # obs = robot.sim.get_sensor_observations(0)
    # rgb = obs["color_sensor"]
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("scr rgb", bgr)
    # cv2.waitKey()

    # tar_hab_tf = cvt_pose_vec2tf(robot.vlmaps_dataloader.base_poses[800])
    # robot.set_agent_state(tar_hab_tf)
    # obs = robot.sim.get_sensor_observations(0)
    # rgb = obs["color_sensor"]
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("tar rgb", bgr)
    # cv2.waitKey()

    # robot.set_agent_state(hab_tf)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # robot.vlmaps_dataloader.from_habitat_tf(tar_hab_tf)
    # tar_row, tar_col, tar_angle_deg = robot.vlmaps_dataloader.to_full_map_pose()
    # robot.empty_recorded_actions()
    # robot.pass_goal_tf([tar_hab_tf])
    # robot.move_to((tar_row, tar_col))

    interactive_map = InteractiveMap(robot.vlmaps_data_save_dirs[config.scene_id], config.map_config)
    rgb_map = robot.map.get_rgb_topdown_map_cropped()
    interactive_map.collect_map_positions(rgb_map)
    tf_hab, agent_state = interactive_map.get_habitat_robot_state(interactive_map.coords[0], interactive_map.coords[1])
    print("habitat_tf: ", tf_hab)

    # TEST move to left
    # robot.set_agent_state(tf_hab)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_to_left("chair")

    # robot.set_agent_state(tf_hab)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_to_right("chair")

    # robot.set_agent_state(tf_hab)
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_south("chair")

    # robot.set_agent_state(tf_hab)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_north("chair")

    # robot.set_agent_state(tf_hab)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_west("chair")

    # robot.set_agent_state(tf_hab)
    # robot.recorded_robot_pos = []
    # robot.recorded_actions_list = []
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_east("chair")

    robot.set_agent_state(tf_hab)
    robot.recorded_robot_pos = []
    robot.recorded_actions_list = []
    rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    cv2.imshow("curr obs", rgb)
    cv2.waitKey()
    robot.turn(90)
    robot.turn(-90)
    robot.move_to_object("counter")

    # robot.set_agent_state(tf_hab)
    # rgb = robot.sim.get_sensor_observations(0)["color_sensor"]
    # cv2.imshow("curr obs", rgb)
    # cv2.waitKey()
    # robot.move_in_between("chair", "sofa")


if __name__ == "__main__":
    main()
