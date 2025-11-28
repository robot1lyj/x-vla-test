import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path for evaluation.* imports.
ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
if REPO_ROOT.name == "evaluation":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.robot.config import CANS
from evaluation.robot.piper.controller import PiperController


def main():
    parser = argparse.ArgumentParser(description="Simple Piper jog without ROS.")
    parser.add_argument("--can", type=str, default=CANS["can_left"], help="CAN interface name.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ctrl = PiperController(can_port=args.can)
    ctrl.enable()
    logging.info("Homing...")
    ctrl.home()
    time.sleep(1.0)

    target = np.array([0.2, 0.0, 0.2, 0.0, 1.57, 0.0])  # x,y,z,roll,pitch,yaw
    logging.info("Moving to pose %s", target)
    ctrl.move_pose(target, gripper=0.0)
    logging.info("Return to home...")
    ctrl.home()
    logging.info("Done. Current state: %s", ctrl.get_state())


if __name__ == "__main__":
    main()
