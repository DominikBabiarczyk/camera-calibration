# Perspective Augmentation (moved)

This folder contains a YAML configuration and a Python script to generate
realistic perspective variations of a chessboard image.

Files:

- `perspective_config.yaml`: parameter ranges used when randomly sampling
  perspective transformation parameters.
- `camera_calibration_config.yaml`: parameter ranges used to sample synthetic
  camera calibration parameters and distortion.
- `augment_perspectives.py`: script that loads the YAML, samples parameters,
  builds a homography or camera-based mesh warp, and writes warped images to an
  output directory.
- `requirements.txt`: minimal Python dependencies.

Quick usage:

```bash
python generate_dataset/creating_various_perspectives/augment_perspectives.py \
  --input path/to/chessboard.png \
  --outdir outputs/augmented \
  --config generate_dataset/creating_various_perspectives/perspective_config.yaml \
  --count 50 \
  --seed 42
```

The script is modular: add more keys to the YAML and extend `sample_params`
and `build_projected_corners` (or camera-specific sampling) to include new transforms.

Camera calibration-style usage:

```bash
python generate_dataset/creating_various_perspectives/augment_perspectives.py \
  --input path/to/chessboard.png \
  --outdir outputs/augmented \
  --config generate_dataset/creating_various_perspectives/camera_calibration_config.yaml \
  --count 50 \
  --seed 42
```
