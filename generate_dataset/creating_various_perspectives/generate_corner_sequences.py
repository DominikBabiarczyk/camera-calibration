"""Generate synthetic corner-only camera-calibration sequences.

This script mirrors the existing camera-style augmentation pipeline, but instead
of saving rendered images it saves the exact point features consumed by the
corner-based neural network.

Each output sequence directory contains:
- corner_sequence.npz: processed features and metadata
- camera_params.yaml: target calibration parameters for the full 9-value vector

The saved `features` array has shape (T, P, 4), where for each detected corner:
- 0: x normalized by image width
- 1: y normalized by image height
- 2: x standardized within the frame
- 3: y standardized within the frame

By default the board geometry matches `generate_dataset/generate_chessboard.py`:
- image size: 640x480
- board: 9x6 squares
- inner corners: 8x5
- square size: 48 px
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from augment_perspectives import (
    load_config,
    project_chessboard_grid,
    sample_camera_dynamic_params_constrained,
    sample_camera_static_params_constrained,
)


VECTOR_ORDER = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]


def build_corner_features(projected_grid: np.ndarray, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert projected board grid to raw inner corners and network features."""
    inner_corners = projected_grid[:-1, :-1, :].reshape(-1, 2).astype(np.float32)

    normalized = inner_corners.copy()
    normalized[:, 0] /= float(image_width)
    normalized[:, 1] /= float(image_height)

    centroid = normalized.mean(axis=0, keepdims=True)
    centered = normalized - centroid
    scale = centered.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    standardized = centered / scale
    features = np.concatenate([normalized, standardized], axis=1).astype(np.float32)
    return inner_corners, features


def build_target(static_params: dict[str, float], image_width: int, image_height: int) -> np.ndarray:
    return np.array(
        [
            static_params["fx"] / float(image_width),
            static_params["fy"] / float(image_height),
            static_params["cx"] / float(image_width),
            static_params["cy"] / float(image_height),
            static_params["k1"],
            static_params["k2"],
            static_params["p1"],
            static_params["p2"],
            static_params["k3"],
        ],
        dtype=np.float32,
    )


def save_camera_params(sequence_outdir: Path, static_params: dict[str, float]) -> None:
    camera_labels = {key: static_params[key] for key in VECTOR_ORDER}
    with open(sequence_outdir / "camera_params.yaml", "w") as f:
        yaml.safe_dump(camera_labels, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corner-only calibration sequences")
    parser.add_argument("--outdir", required=True, help="Output directory for generated sequences")
    parser.add_argument(
        "--config",
        default="generate_dataset/creating_various_perspectives/camera_calibration_config.yaml",
        help="Camera calibration config YAML",
    )
    parser.add_argument("--count", type=int, default=None, help="Frames per sequence")
    parser.add_argument("--sequences", type=int, default=None, help="Number of sequences to generate")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--board-cols", type=int, default=8, help="Number of inner corners in x direction")
    parser.add_argument("--board-rows", type=int, default=5, help="Number of inner corners in y direction")
    parser.add_argument("--square-size", type=float, default=48.0, help="Checkerboard square size in pixels")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("mode") != "camera":
        raise ValueError("This generator requires a config with mode: camera.")

    count = args.count if args.count is not None else int(cfg.get("samples", 50))
    sequences = args.sequences if args.sequences is not None else int(cfg.get("sequences", 1))
    seed = args.seed if args.seed is not None else cfg.get("seed")
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for seq_idx in range(sequences):
        sequence_outdir = outdir / f"sequence_{seq_idx:03d}"
        sequence_outdir.mkdir(parents=True, exist_ok=True)

        static_params = sample_camera_static_params_constrained(cfg, args.width, args.height, rng)
        static_params["board_cols"] = args.board_cols
        static_params["board_rows"] = args.board_rows
        static_params["square_size"] = args.square_size
        save_camera_params(sequence_outdir, static_params)

        features_sequence: list[np.ndarray] = []
        raw_corners_sequence: list[np.ndarray] = []

        for _ in range(count):
            frame_params = sample_camera_dynamic_params_constrained(
                cfg,
                args.width,
                args.height,
                rng,
                static_params,
            )
            _, projected_grid = project_chessboard_grid(args.width, args.height, frame_params)
            raw_corners, features = build_corner_features(projected_grid, args.width, args.height)
            raw_corners_sequence.append(raw_corners)
            features_sequence.append(features)

        np.savez_compressed(
            sequence_outdir / "corner_sequence.npz",
            features=np.stack(features_sequence, axis=0).astype(np.float32),
            raw_corners=np.stack(raw_corners_sequence, axis=0).astype(np.float32),
            target=build_target(static_params, args.width, args.height),
            image_width=np.array(args.width, dtype=np.int32),
            image_height=np.array(args.height, dtype=np.int32),
            vector_order=np.array(VECTOR_ORDER),
            board_cols=np.array(static_params["board_cols"], dtype=np.int32),
            board_rows=np.array(static_params["board_rows"], dtype=np.int32),
            square_size=np.array(static_params["square_size"], dtype=np.float32),
        )

        print(f"Saved: {sequence_outdir}")


if __name__ == "__main__":
    main()