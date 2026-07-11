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


def create_chessboard_image() -> np.ndarray:
    """Create a single synthetic chessboard image."""
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 255, dtype=np.uint8)

    for row in range(NUM_SQUARES_Y):
        for col in range(NUM_SQUARES_X):
            x0 = BOARD_ORIGIN_X + col * SQUARE_SIZE_PX
            y0 = BOARD_ORIGIN_Y + row * SQUARE_SIZE_PX
            x1 = x0 + SQUARE_SIZE_PX
            y1 = y0 + SQUARE_SIZE_PX

            if (row + col) % 2 == 0:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)

            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)

    return image


def save_chessboard_images(count: int = NUM_IMAGES) -> None:
    """Save a batch of synthetic chessboard images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        image = create_chessboard_image()
        path = OUTPUT_DIR / f"chessboard_{i:03d}.png"
        cv2.imwrite(str(path), image)
        print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating synthetic chessboard images...")
    print(
        f"Board: {NUM_SQUARES_X}x{NUM_SQUARES_Y} squares, {SQUARE_SIZE_PX}px per square"
    )
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    save_chessboard_images()
