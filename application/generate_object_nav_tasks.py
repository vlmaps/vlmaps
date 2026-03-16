#!/usr/bin/env python3
"""
generate_object_nav_tasks.py
============================
Generates ``object_navigation_tasks.json`` for every scene folder found in
the VLMAPS dataset directory.

Each JSON file is a *list* of task dictionaries with the structure expected by
``HabitatObjectNavigationTask.load_task()`` / ``setup_task()``:

    [
        {
            "task_id"       : int,          # index in the list
            "tf_habitat"    : [float×16],   # row-major 4×4 habitat TF
            "map_grid_size" : int,          # gs  (cells per side)
            "map_cell_size" : float,        # cs  (metres per voxel)
            "scene"         : str,          # e.g. "00800-TEEsavR23oF"
            "instruction"   : str,          # natural-language navigation goal
            "objects_info"  : [{"name": str}, ...]
        },
        ...
    ]

Usage (from the repository root)
---------------------------------
    python application/generate_object_nav_tasks.py

Overrides can be passed as Hydra CLI arguments, e.g.:
    python application/generate_object_nav_tasks.py nav.tasks_per_scene=20

The script is intentionally lightweight: it reads poses.txt from each scene
folder directly, without initialising the Habitat simulator.  This makes it
fast and dependency-free for task *generation*.  The heavyweight robot setup
is only required when *running* the evaluation.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from vlmaps.utils.mapping_utils import cvt_pose_vec2tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indoor object categories drawn from the Matterport3D / HM3D ontology.
# These labels must match the strings returned by
# ``habitat_sim.SemanticObject.category.name()`` during evaluation.
TARGET_CATEGORIES: List[str] = [
    "chair",
    "table",
    "sofa",
    "bed",
    "sink",
    "toilet",
    "cabinet",
]

# Default number of sequential sub-goals (objects) per task.
# The evaluation checks them one by one in order, so keep this small (2–3).
DEFAULT_SUBGOALS_PER_TASK: int = 2

# Default tasks to generate per scene if not set in the config.
DEFAULT_TASKS_PER_SCENE: int = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_instruction(categories: List[str]) -> str:
    """
    Turn a list of object category names into a natural-language instruction
    that ``parse_object_goal_instruction()`` (GPT-4) can parse back cleanly.

    Examples
    --------
    ["chair"]              -> "Go to the chair."
    ["sofa", "table"]      -> "Go to the sofa, then go to the table."
    ["bed", "sink", "chair"] ->
        "Go to the bed, then go to the sink, then go to the chair."
    """
    if len(categories) == 1:
        return f"Go to the {categories[0]}."
    steps = [f"go to the {c}" for c in categories]
    # Capitalise only the very first word.
    steps[0] = steps[0][0].upper() + steps[0][1:]
    return ", then ".join(steps) + "."


def load_base_poses(scene_dir: Path) -> np.ndarray:
    """
    Load robot base poses from ``<scene_dir>/poses.txt``.

    Each line contains 7 space/tab-separated floats:
        px  py  pz  qx  qy  qz  qw   (position + quaternion, XYZW order)

    Returns
    -------
    np.ndarray of shape (N, 7)
    """
    pose_path = scene_dir / "poses.txt"
    if not pose_path.exists():
        raise FileNotFoundError(
            f"poses.txt not found in {scene_dir}. "
            "Has the data-collection step been run for this scene?"
        )
    poses = np.loadtxt(pose_path)
    if poses.ndim == 1:
        # Single-pose edge case: promote to (1, 7)
        poses = poses[np.newaxis, :]
    if poses.shape[1] != 7:
        raise ValueError(
            f"Expected 7 columns in poses.txt (px py pz qx qy qz qw), "
            f"got {poses.shape[1]} in {pose_path}"
        )
    return poses


def sample_pose_indices(
    n_poses: int,
    n_tasks: int,
    rng: random.Random,
) -> List[int]:
    """
    Choose ``n_tasks`` pose indices spread across the full trajectory.

    Strategy
    --------
    1. Build a strided candidate list to ensure spatial diversity.
    2. Shuffle it with the supplied RNG for reproducible randomness.
    3. Return at most ``n_tasks`` indices in sorted order.
    """
    stride = max(1, n_poses // n_tasks)
    candidates = list(range(0, n_poses, stride))

    # If the stride gave us fewer than needed, fall back to all poses.
    if len(candidates) < n_tasks:
        candidates = list(range(n_poses))

    rng.shuffle(candidates)
    selected = sorted(candidates[:n_tasks])
    return selected


def generate_tasks_for_scene(
    scene_dir: Path,
    scene_name: str,
    gs: int,
    cs: float,
    n_tasks: int,
    n_subgoals: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Generate the task list for a single scene.

    Parameters
    ----------
    scene_dir   : Path to the scene data folder (contains poses.txt).
    scene_name  : Habitat scene identifier, e.g. "00800-TEEsavR23oF".
    gs          : Map grid size (number of cells per side).
    cs          : Map cell size (metres per voxel).
    n_tasks     : Number of tasks to generate.
    n_subgoals  : Number of object sub-goals per task.
    rng         : Seeded random.Random instance for reproducibility.

    Returns
    -------
    List of task dictionaries ready for json.dump().
    """
    base_poses = load_base_poses(scene_dir)
    n_poses = len(base_poses)

    # Clamp the requested task count to the available poses.
    effective_tasks = min(n_tasks, n_poses)
    if effective_tasks < n_tasks:
        print(
            f"  [warn] Only {n_poses} poses available; "
            f"generating {effective_tasks} tasks (requested {n_tasks})."
        )

    pose_indices = sample_pose_indices(n_poses, effective_tasks, rng)

    tasks: List[Dict[str, Any]] = []
    for task_id, pose_idx in enumerate(pose_indices):
        # Convert 7-D pose vector → 4×4 habitat transformation matrix.
        tf_hab: np.ndarray = cvt_pose_vec2tf(base_poses[pose_idx])

        # Pick n_subgoals distinct categories at random.
        n_pick = min(n_subgoals, len(TARGET_CATEGORIES))
        categories: List[str] = rng.sample(TARGET_CATEGORIES, n_pick)

        task: Dict[str, Any] = {
            "task_id": task_id,
            # Store as a flat 16-element list; setup_task() does .reshape((4,4))
            "tf_habitat": tf_hab.flatten().tolist(),
            "map_grid_size": gs,
            "map_cell_size": cs,
            "scene": scene_name,
            "instruction": build_instruction(categories),
            "objects_info": [{"name": c} for c in categories],
        }
        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Main entry point (Hydra)
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../config",          # relative to this file's location
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    # ── Configuration ────────────────────────────────────────────────────────
    gs: int   = int(config.params.gs)
    cs: float = float(config.params.cs)

    # tasks_per_scene can be set in object_goal_navigation_cfg.yaml under nav:
    n_tasks: int = int(
        getattr(config.nav, "tasks_per_scene", DEFAULT_TASKS_PER_SCENE)
    )
    n_subgoals: int = DEFAULT_SUBGOALS_PER_TASK

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    if not data_dir.exists():
        print(f"ERROR: dataset directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Ignore macOS artefact folders (.DS_Store etc.)
    scene_dirs: List[Path] = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not scene_dirs:
        print(f"ERROR: no scene folders found inside {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Fixed seed → identical task files across runs; change to get new sets.
    rng = random.Random(42)

    print(f"\nGenerating object navigation tasks")
    print(f"  Dataset directory : {data_dir}")
    print(f"  Scenes found      : {len(scene_dirs)}")
    print(f"  Tasks per scene   : {n_tasks}")
    print(f"  Sub-goals/task    : {n_subgoals}")
    print(f"  Map gs / cs       : {gs} / {cs}")
    print(f"  Target categories : {TARGET_CATEGORIES}\n")

    # ── Per-scene generation ─────────────────────────────────────────────────
    generated = 0
    skipped   = 0

    for scene_dir in scene_dirs:
        output_path = scene_dir / "object_navigation_tasks.json"

        if output_path.exists():
            print(f"  [skip]  {scene_dir.name}  (file already exists)")
            skipped += 1
            continue

        # Strip the trailing "_<N>" index to get the Habitat scene identifier.
        # e.g.  "00800-TEEsavR23oF_1"  →  "00800-TEEsavR23oF"
        scene_name = scene_dir.name.rsplit("_", 1)[0]

        print(f"  [gen]   {scene_dir.name}")

        try:
            tasks = generate_tasks_for_scene(
                scene_dir, scene_name, gs, cs, n_tasks, n_subgoals, rng
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"          WARNING: {exc}", file=sys.stderr)
            continue

        with open(output_path, "w") as fh:
            json.dump(tasks, fh, indent=2)

        print(f"          Wrote {len(tasks)} tasks → {output_path}")
        generated += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\nDone.  Generated: {generated} file(s),  skipped: {skipped} file(s).")


if __name__ == "__main__":
    main()
