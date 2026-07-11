import shutil
import subprocess
from pathlib import Path

# Resolve paths relative to the project root (two levels above this file)
base_dir = Path(__file__).resolve().parents[2]
video_path = base_dir / "data" / "00046.MTS"
output_dir = base_dir / "data" / "sequences"

print("Resolved video path:", video_path)
print("Resolved output dir:", output_dir)
print("Video exists:", video_path.exists())

if not video_path.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku wideo: {video_path}")

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    raise RuntimeError("Nie znaleziono programu 'ffmpeg' w systemie.")

output_dir.mkdir(parents=True, exist_ok=True)

# Usuń stare klatki, aby wynik odpowiadał bieżącemu uruchomieniu.
for existing_frame in output_dir.glob("frame_*.jpg"):
    existing_frame.unlink()

output_pattern = output_dir / "frame_%06d.jpg"

command = [
    ffmpeg_path,
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    str(video_path),
    "-vsync",
    "0",
    "-start_number",
    "0",
    "-q:v",
    "2",
    str(output_pattern),
]

print("Using ffmpeg:", ffmpeg_path)
subprocess.run(command, check=True)

frame_count = len(list(output_dir.glob("frame_*.jpg")))
print("Gotowe! Zapisano klatki:", frame_count)