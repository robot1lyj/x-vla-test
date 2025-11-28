"""
ROS-free evaluation loop for SoftFold-Agilex:
- Grabs multi-camera frames via OpenCV
- Queries X-VLA HTTP server
- Sends poses to Piper through custom IK
"""
import argparse
import collections
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import json_numpy
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

# Ensure repo root on sys.path for evaluation.* imports.
ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
# If we resolved to /evaluation, go one level up to repo root.
if REPO_ROOT.name == "evaluation":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.utils.rotation import abs_6d_2_abs_euler, rotation_matrix_to_6d
from evaluation.robot.camera.camera_manager import CameraManager
from evaluation.robot.config import CAMERAS, MOTION, SERVER, CANS
from evaluation.robot.piper.controller import PiperController


def build_proprio(eef_xyzrpy: np.ndarray, gripper: float) -> np.ndarray:
    """Pack single-arm eef into dual-arm format expected by the model (right arm zeros)."""
    rot = R.from_euler("xyz", eef_xyzrpy[3:], degrees=False).as_matrix()
    rot6d = rotation_matrix_to_6d(rot)
    left = np.concatenate([eef_xyzrpy[:3], rot6d, [gripper]], axis=0)
    right = np.zeros_like(left)
    return np.concatenate([left, right], axis=0)  # shape (20,)


class XVLAHttpClient:
    def __init__(self, host: str, port: int, chunk_size: int):
        self.url = f"http://{host}:{port}/act"
        self.chunk_size = chunk_size
        self.action_plan = collections.deque()

    def reset(self):
        self.action_plan.clear()

    def predict(self, obs: Dict) -> np.ndarray:
        if not self.action_plan:
            payload = {
                "proprio": json_numpy.dumps(obs["proprio"]),
                "image0": json_numpy.dumps(obs["images"]["cam_high"]),
                "image1": json_numpy.dumps(obs["images"]["cam_left_wrist"]),
                "image2": json_numpy.dumps(obs["images"]["cam_right_wrist"]),
                "language_instruction": obs.get("language", "flatten the cloth and fold it"),
                "steps": 10,
                "domain_id": obs.get("domain_id", SERVER["domain_id"]),
            }
            resp = requests.post(self.url, json=payload, timeout=10)
            resp.raise_for_status()
            action = resp.json()["action"]
            self.action_plan.extend(action[: self.chunk_size])
        return np.asarray(self.action_plan.popleft(), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run SoftFold-Agilex without ROS.")
    parser.add_argument("--host", type=str, default=SERVER["host"])
    parser.add_argument("--port", type=int, default=SERVER["port"])
    parser.add_argument("--chunk_size", type=int, default=SERVER["chunk_size"])
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--language", type=str, default="flatten the cloth and then fold it")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cam_mgr = CameraManager(CAMERAS)
    cam_mgr.open_all()
    ctrl = PiperController(can_port=CANS["can_left"])
    ctrl.enable()
    client = XVLAHttpClient(args.host, args.port, args.chunk_size)

    rate = 1.0 / MOTION["publish_rate"]
    try:
        for step in range(args.max_steps):
            t0 = time.time()
            frames = cam_mgr.read_all()
            state = ctrl.get_state()
            proprio = build_proprio(state["eef"], state["qpos"][-1])
            obs = {
                "proprio": proprio,
                "images": frames,
                "language": args.language,
                "domain_id": SERVER["domain_id"],
            }
            raw_action = client.predict(obs)
            action = abs_6d_2_abs_euler(raw_action)
            left = action[:7]
            ctrl.move_pose(np.array(left[:6]), gripper=left[6])
            elapsed = time.time() - t0
            sleep_time = max(0.0, rate - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cam_mgr.release_all()
        ctrl.disable()


if __name__ == "__main__":
    main()
