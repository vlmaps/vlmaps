import os
import time
import hydra
import numpy as np
from omegaconf import DictConfig
import habitat_sim
from vlmaps.utils.habitat_utils import *


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collect_dataset.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.makedirs(config.data_paths.vlmaps_data_dir, exist_ok=True)
    dataset_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"

    scene_dirs = []
    for scene_name in config.scene_names:
        id = 1
        while True:
            scene_dir = dataset_dir / f"{scene_name}_{id}"
            if not scene_dir.exists():
                break
            id += 1
        print(f"Collecting data for scene {scene_name}")
        print(f"Data will be saved at {scene_dir}")
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_dirs.append(scene_dir)

        # test_scene_dir = config.data_paths.habitat_scene_dir
        # # test_scene_dir = "/home/hcg/hcg/phd/projects/vln/data/scene_datasets/mp3d/v1/tasks/mp3d_habitat/mp3d/"
        # # img_save_dir = "/home/huang/Pictures/vln"
        # img_save_dir = "/home/huang/hcg/projects/vln/data/clip_mapping/description_videos/"
        # # img_save_dir = "/home/hcg/Pictures/vln_diff_size_images"
        # scenes_names = os.listdir(test_scene_dir)
        # SCENE_ID = 0  # random.randint(0, len(scenes_names))
        # assert SCENE_ID < len(scenes_names) - 1

        # img_save_dir += f"{scenes_names[SCENE_ID]}_1"
        # os.makedirs(img_save_dir, exist_ok=True)

        test_scene = os.path.join(config.data_paths.habitat_scene_dir, scene_name, scene_name + ".glb")

        sim_setting = {
            "scene": test_scene,
            "default_agent": 0,
            "sensor_height": 1.5,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "lidar_sensor": True,
            "move_forward": 0.1,
            "turn_left": 5,
            "turn_right": 5,
            "width": 1080,
            "height": 720,
            "enable_physics": False,
            "seed": 42,
            "lidar_fov": 360,
            "depth_img_for_lidar_n": 20,
            "img_save_dir": scene_dir,
        }

        # cfg = make_simple_cfg(sim_setting)
        cfg = make_cfg(sim_setting)

        # create a simulator instance
        sim = habitat_sim.Simulator(cfg)
        scene = sim.semantic_scene
        objs = scene.objects
        levels = scene.levels
        for level in levels:
            print(level.id, level.aabb.center, level.aabb.sizes)
            print(
                level.id, level.aabb.center[1] - level.aabb.sizes[1] / 2, level.aabb.center[1] + level.aabb.sizes[1] / 2
            )
        # for obj in objs:
        #     print(obj.id, obj.region.category.name(), obj.category.name(), obj.obb.center, obj.obb.sizes)
        obj2cls = {int(obj.id.split("_")[-1]): (obj.category.index(), obj.category.name()) for obj in scene.objects}

        # initialize the agent
        agent = sim.initialize_agent(sim_setting["default_agent"])

        agent_state = habitat_sim.AgentState()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        # random_pt = sim.pathfinder.get_random_navigable_point()
        # agent_state.position = np.array([1.5, height_list[np.random.randint(0, len(height_list) - 1)], 4.0])
        agent_state.position = random_pt
        agent.set_state(agent_state)

        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        init_agent_state = agent_state
        actions_list = []

        obs = sim.get_sensor_observations(0)
        last_action = None
        release_count = 0
        while True:
            show_rgb(obs)
            k, action = keyboard_control_fast()
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = sim.get_agent(0).get_state()
                    actions_list = []
                    continue
                last_action = action
                release_count = 0
            else:
                if last_action is None:
                    time.sleep(0.01)
                    continue
                else:
                    release_count += 1
                    if release_count > 1:
                        print("stop after release")
                        last_action = None
                        release_count = 0
                        continue
                    action = last_action

            obs = sim.step(action)
            actions_list.append(action)

        actions_list = [x for x in actions_list if x != "pause"]

        agent_states = []
        agent.set_state(init_agent_state)
        obs = sim.get_sensor_observations(0)
        root_save_dir = sim_setting["img_save_dir"]
        save_obs(root_save_dir, sim_setting, obs, 0, obj2cls)
        # save_state(root_save_dir, sim_setting, agent.get_state(), 0)
        agent_states.append(agent.get_state())

        print(f"saving frame 0/{len(actions_list) + 1}...")

        for action_i, action in enumerate(actions_list):
            obs = sim.step(action)
            agent = sim.get_agent(0)
            print(f"saving frame {action_i + 1}/{len(actions_list) + 1}...")
            save_obs(root_save_dir, sim_setting, obs, action_i + 1, obj2cls)
            agent_states.append(agent.get_state())
        save_states(root_save_dir, agent_states)


if __name__ == "__main__":

    main()
