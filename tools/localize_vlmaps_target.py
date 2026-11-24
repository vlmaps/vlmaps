#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from hydra import compose, initialize_config_dir

from vlmaps.map.vlmap import VLMap


def load_config(repo):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(repo / "config"),
    ):
        return compose(
            config_name="map_indexing_cfg",
            overrides=[
                "data_paths=jackal_benchmark_build",
                "map_config=vlmaps_jackal",
                "params=default",
                "init_categories=false",
            ],
        )


def connected_components(metric, grid):
    xy = grid[:, [0, 1]]

    origin = xy.min(axis=0)
    shifted = xy - origin

    h = int(shifted[:, 0].max()) + 1
    w = int(shifted[:, 1].max()) + 1

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[
        shifted[:, 0],
        shifted[:, 1],
    ] = 1

    nlabels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    voxel_labels = labels[
        shifted[:, 0],
        shifted[:, 1],
    ]

    components = []

    for label in range(1, nlabels):
        selected = voxel_labels == label
        points = metric[selected]

        if len(points) == 0:
            continue

        center = np.median(points, axis=0)

        components.append({
            "points": points,
            "center": center,
            "cells": int(
                stats[label, cv2.CC_STAT_AREA]
            ),
        })

    return components


def merge_components(components, distance_m):
    """
    Same conceptual rule used in the CLIPSeg benchmark:
    nearby semantic fragments belong to one physical instance.
    """

    n = len(components)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            a = components[i]["center"][:2]
            b = components[j]["center"][:2]

            if np.linalg.norm(a - b) <= distance_m:
                union(i, j)

    groups = {}

    for i in range(n):
        groups.setdefault(
            find(i),
            [],
        ).append(i)

    instances = []

    for indexes in groups.values():
        points = np.vstack([
            components[i]["points"]
            for i in indexes
        ])

        center = np.median(
            points,
            axis=0,
        )

        instances.append({
            "component_ids": indexes,
            "points": points,
            "center": center,
            "voxels": int(len(points)),
        })

    return instances


def select_nearest_instance(instances, start_xy):
    best = None
    best_distance = float("inf")

    for instance_id, instance in enumerate(instances):
        center = instance["center"]

        distance = math.hypot(
            center[0] - start_xy[0],
            center[1] - start_xy[1],
        )

        if distance < best_distance:
            best_distance = distance
            best = (
                instance_id,
                instance,
                distance,
            )

    return best


def make_goal(start_xy, target_xy, standoff):
    dx = target_xy[0] - start_xy[0]
    dy = target_xy[1] - start_xy[1]

    distance = math.hypot(dx, dy)

    if distance <= standoff + 0.05:
        return None

    goal_x = (
        target_xy[0]
        - standoff * dx / distance
    )

    goal_y = (
        target_xy[1]
        - standoff * dy / distance
    )

    yaw = math.atan2(
        target_xy[1] - goal_y,
        target_xy[0] - goal_x,
    )

    return {
        "x": float(goal_x),
        "y": float(goal_y),
        "yaw": float(yaw),
        "qz": float(math.sin(yaw / 2.0)),
        "qw": float(math.cos(yaw / 2.0)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", required=True)
    parser.add_argument("--query", required=True)

    parser.add_argument(
        "--positions-file",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--instance-merge-distance",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--standoff",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()

    repo = Path.cwd()

    scene = (
        Path.home()
        / "vlmaps_data_benchmark"
        / args.scenario
    )

    if not (
        scene
        / "vlmap_cam"
        / "vlmaps_cam.h5df"
    ).exists():
        raise RuntimeError(
            f"VLMap not found for {args.scenario}"
        )

    cfg = load_config(repo)

    vlmap = VLMap(
        cfg.map_config,
        data_dir=scene,
    )

    vlmap.load_map(scene)
    vlmap._init_clip()

    mask = np.asarray(
        vlmap.index_map(
            args.query,
            with_init_cat=False,
        )
    ).reshape(-1)

    positive = np.flatnonzero(mask)

    if len(positive) == 0:
        raise RuntimeError(
            f"No semantic voxels for query "
            f"'{args.query}'"
        )

    grid = np.asarray(
        vlmap.grid_pos[positive],
        dtype=np.int32,
    )

    metric = (
        np.asarray(
            vlmap.pcd_min,
            dtype=float,
        )
        + grid.astype(float)
        * float(vlmap.cs)
    )

    components = connected_components(
        metric,
        grid,
    )

    instances = merge_components(
        components,
        args.instance_merge_distance,
    )

    positions_path = Path(
        args.positions_file
    ).expanduser()

    with positions_path.open() as f:
        positions = (
            yaml.safe_load(f)["positions"]
        )

    result = {
        "scenario": args.scenario,
        "query": args.query,
        "frame": "odom",
        "positive_voxels": int(
            len(positive)
        ),
        "component_count": int(
            len(components)
        ),
        "instance_count": int(
            len(instances)
        ),
        "instance_merge_distance_m":
            float(
                args.instance_merge_distance
            ),
        "standoff_distance_m":
            float(args.standoff),
        "episodes": {},
    }

    print(
        f"positive voxels: {len(positive)}"
    )

    print(
        f"components: {len(components)}"
    )

    print(
        f"merged instances: {len(instances)}"
    )

    for position_id, pose in positions.items():
        start_xy = np.array([
            float(pose["x"]),
            float(pose["y"]),
        ])

        selected = select_nearest_instance(
            instances,
            start_xy,
        )

        if selected is None:
            raise RuntimeError(
                f"No instance for {position_id}"
            )

        (
            instance_id,
            instance,
            distance,
        ) = selected

        center = instance["center"]

        goal = make_goal(
            start_xy,
            center[:2],
            args.standoff,
        )

        result["episodes"][
            position_id
        ] = {
            "start": {
                "x": float(start_xy[0]),
                "y": float(start_xy[1]),
            },
            "selected_instance":
                int(instance_id),
            "target": {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            },
            "distance_to_target_m":
                float(distance),
            "goal": goal,
        }

        print()
        print(position_id)

        print(
            "  target:",
            f"({center[0]:.3f}, "
            f"{center[1]:.3f}, "
            f"{center[2]:.3f})",
        )

        print(
            "  distance:",
            f"{distance:.3f} m",
        )

        if goal is not None:
            print(
                "  goal:",
                f"({goal['x']:.3f}, "
                f"{goal['y']:.3f})",
            )

    output = Path(
        args.output
    ).expanduser()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w") as f:
        yaml.safe_dump(
            result,
            f,
            sort_keys=False,
        )

    print()
    print("Saved:")
    print(output)


if __name__ == "__main__":
    main()
