"""Generate synthetic chessboard images with hardcoded board dimensions.

This script creates a set of synthetic checkerboard images saved under
`generate_dataset/boards/`.

The board size is hardcoded, as requested:
- NUM_SQUARES_X = 9 (squares along width)
- NUM_SQUARES_Y = 6 (squares along height)
- SQUARE_SIZE_PX = 48 (square size in pixels)

The board is centered in a fixed image size of 640x480.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "boards"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_SQUARES_X = 9
NUM_SQUARES_Y = 6
SQUARE_SIZE_PX = 48

# The generated board dimensions in pixels.
BOARD_WIDTH = NUM_SQUARES_X * SQUARE_SIZE_PX
BOARD_HEIGHT = NUM_SQUARES_Y * SQUARE_SIZE_PX

# Center the board in the image.
BOARD_ORIGIN_X = (IMAGE_WIDTH - BOARD_WIDTH) // 2
BOARD_ORIGIN_Y = (IMAGE_HEIGHT - BOARD_HEIGHT) // 2

NUM_IMAGES = 1


def create_chessboard_image(
    num_squares_x: int = NUM_SQUARES_X,
    num_squares_y: int = NUM_SQUARES_Y,
    square_size_px: int = SQUARE_SIZE_PX,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> np.ndarray:
    """Create a single synthetic chessboard image."""
    image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    board_width = num_squares_x * square_size_px
    board_height = num_squares_y * square_size_px
    origin_x = (image_width - board_width) // 2
    origin_y = (image_height - board_height) // 2

    for row in range(num_squares_y):
        for col in range(num_squares_x):
            x0 = origin_x + col * square_size_px
            y0 = origin_y + row * square_size_px
            x1 = x0 + square_size_px
            y1 = y0 + square_size_px

            if (row + col) % 2 == 0:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)

            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)

    return image


def save_chessboard_images(
    count: int = NUM_IMAGES,
    num_squares_x: int = NUM_SQUARES_X,
    num_squares_y: int = NUM_SQUARES_Y,
    square_size_px: int = SQUARE_SIZE_PX,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> None:
    """Save a batch of synthetic chessboard images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image = create_chessboard_image(
            num_squares_x, num_squares_y, square_size_px, image_width, image_height
        )
        path = OUTPUT_DIR / f"chessboard_{i:03d}.png"
        cv2.imwrite(str(path), image)
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic chessboard image")
    parser.add_argument("--squares-x", type=int, default=NUM_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=NUM_SQUARES_Y)
    parser.add_argument("--square-size", type=int, default=SQUARE_SIZE_PX)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    print("Generating synthetic chessboard images...")
    print(
        f"Board: {args.squares_x}x{args.squares_y} squares, {args.square_size}px per square"
    )
    print(f"Image size: {args.width}x{args.height}")
    if args.output_name is None:
        save_chessboard_images(
            1, args.squares_x, args.squares_y, args.square_size, args.width, args.height
        )
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(OUTPUT_DIR / args.output_name),
            create_chessboard_image(
                args.squares_x, args.squares_y, args.square_size, args.width, args.height
            ),
        )
