import os
import cv2
import time
from pathlib import Path

import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R


def make_simple_cfg(scene_path: str, save_dir: str):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [720, 1080]
    color_sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [720, 1080]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_specs.append(depth_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def quat_xyzw_from_habitat(quat) -> np.ndarray:
    return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)


def save_frame(obs, state, save_dir: Path, frame_id: int, poses_list):
    rgb_dir = save_dir / "rgb"
    depth_dir = save_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb = obs["color_sensor"]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_dir / f"{frame_id:06}.png"), rgb_bgr)

    depth = obs["depth_sensor"]
    np.save(depth_dir / f"{frame_id:06}.npy", depth)

    pos = state.position
    quat = quat_xyzw_from_habitat(state.rotation)
    poses_list.append([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])


@hydra.main(version_base=None, config_path="../config", config_name="collect_dataset.yaml")
def main(config: DictConfig):
    scene_name = config.scene_names[0]
    scene_path = os.path.join(
        config.data_paths.habitat_scene_dir,
        scene_name,
        scene_name + ".glb",
    )

    dataset_root = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    idx = 1
    while (dataset_root / f"{scene_name}_{idx}").exists():
        idx += 1
    save_dir = dataset_root / f"{scene_name}_{idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {scene_path}")
    print(f"Saving to: {save_dir}")

    cfg = make_simple_cfg(scene_path, str(save_dir))
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)

    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)

    poses_list = []
    frame_id = 0

    print("")
    print("Controles:")
    print("  w: avanzar")
    print("  a: girar izquierda")
    print("  d: girar derecha")
    print("  s: guardar frame actual")
    print("  q: salir y guardar poses")
    print("")

    while True:
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"]
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", rgb_bgr)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("w"):
            agent.act("move_forward")
        elif key == ord("a"):
            agent.act("turn_left")
        elif key == ord("d"):
            agent.act("turn_right")
        elif key == ord("s"):
            obs = sim.get_sensor_observations()
            state = agent.get_state()
            save_frame(obs, state, save_dir, frame_id, poses_list)
            print(f"Guardado frame {frame_id:06}")
            frame_id += 1
        elif key == ord("q"):
            break

    if poses_list:
        poses = np.array(poses_list, dtype=np.float32)
        np.savetxt(save_dir / "poses.txt", poses)
        print(f"Guardadas {len(poses_list)} poses en {save_dir / 'poses.txt'}")
    else:
        print("No se guardó ningún frame")

    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
