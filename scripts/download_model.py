from pathlib import Path
from urllib.request import urlopen


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.is_file():
        print(f"Already exists: {DESTINATION}")
        return

    temporary = DESTINATION.with_suffix(".task.part")
    print(f"Downloading MediaPipe model to {DESTINATION}...")
    with urlopen(MODEL_URL) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(DESTINATION)
    print("done")


if __name__ == "__main__":
    main()

