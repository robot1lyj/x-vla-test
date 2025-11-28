import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure repo root on sys.path for evaluation.* imports.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.robot.config import CAMERAS
from evaluation.robot.camera.camera_manager import CameraManager


def main():
    parser = argparse.ArgumentParser(description="Preview multiple cameras without ROS.")
    parser.add_argument("--duration", type=float, default=100, help="Preview seconds.")
    parser.add_argument("--window", type=str, default="cams", help="OpenCV window name.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mgr = CameraManager(CAMERAS)
    mgr.open_all()
    start = time.time()
    try:
        while time.time() - start < args.duration:
            frames = mgr.read_all()

            # Ordered display: high, left wrist, right wrist.
            names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            tiles = []
            for name in names:
                frame = frames[name]
                cfg = CAMERAS[name]
                frame = cv2.resize(frame, (cfg["width"], cfg["height"]))
                cv2.putText(frame, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                tiles.append(frame)
            canvas = np.hstack(tiles)
            cv2.imshow(args.window, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        mgr.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
