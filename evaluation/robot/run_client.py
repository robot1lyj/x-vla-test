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
from typing import Dict, Optional

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

from evaluation.robot.camera.camera_manager import CameraManager
from evaluation.robot.config import CAMERAS, CONTROL_MODE, MOTION, SERVER, CANS
from evaluation.robot.piper.controller import PiperController
from evaluation.robot.rotation import abs_6d_2_abs_euler, rotation_matrix_to_6d


def build_proprio(
    left_eef_xyzrpy: np.ndarray,
    left_gripper: float,
    right_eef_xyzrpy: Optional[np.ndarray] = None,
    right_gripper: float = 0.0,
) -> np.ndarray:
    """Pack dual-arm eef (xyzrpy + gripper) into 20-dim (left+right)."""
    rot_l = R.from_euler("xyz", left_eef_xyzrpy[3:], degrees=False).as_matrix()
    rot6d_l = rotation_matrix_to_6d(rot_l)
    left = np.concatenate([left_eef_xyzrpy[:3], rot6d_l, [left_gripper]], axis=0)

    if right_eef_xyzrpy is None:
        right = np.zeros_like(left)
    else:
        rot_r = R.from_euler("xyz", right_eef_xyzrpy[3:], degrees=False).as_matrix()
        rot6d_r = rotation_matrix_to_6d(rot_r)
        right = np.concatenate([right_eef_xyzrpy[:3], rot6d_r, [right_gripper]], axis=0)
    return np.concatenate([left, right], axis=0)


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


def split_action(raw_action: np.ndarray) -> Dict[str, np.ndarray]:
    """
    SoftFold-Agilex: model输出为 eef 6d (20) 或关节角（14，兼容）。
    """
    if raw_action.shape[0] == 20:
        eef = abs_6d_2_abs_euler(raw_action)
        left = eef[:7]
        right = eef[7:]
        return {"left_eef": left, "right_eef": right}
    if raw_action.shape[0] == 14:
        left = raw_action[:7]
        right = raw_action[7:]
        return {"left_joint": left, "right_joint": right}
    raise ValueError(f"Unexpected action length {raw_action.shape[0]} (expect 14 or 20)")


def main():
    parser = argparse.ArgumentParser(description="Run SoftFold-Agilex without ROS.")
    parser.add_argument("--host", type=str, default=SERVER["host"])
    parser.add_argument("--port", type=int, default=SERVER["port"])
    parser.add_argument("--chunk_size", type=int, default=SERVER["chunk_size"])
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--language", type=str, default="flatten the cloth and then fold it")
    parser.add_argument("--mode", type=str, default=CONTROL_MODE, choices=["eef", "joint"], help="eef (6D) or joint control")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cam_mgr = CameraManager(CAMERAS)
    cam_mgr.open_all()
    ctrl_left = PiperController(can_port=CANS["can_left"])
    ctrl_right: Optional[PiperController] = None
    if "can_right" in CANS and CANS["can_right"]:
        ctrl_right = PiperController(can_port=CANS["can_right"])
    ctrl.enable()
    if ctrl_right:
        ctrl_right.enable()
    client = XVLAHttpClient(args.host, args.port, args.chunk_size)

    rate = 1.0 / MOTION["publish_rate"]
    try:
        for step in range(args.max_steps):
            t0 = time.time()
            frames = cam_mgr.read_all()
            state_l = ctrl_left.get_state()
            eef_l = state_l["eef"] if state_l["eef"] is not None else np.zeros(6)
            qpos_l = state_l["qpos"]

            eef_r = None
            grip_r = 0.0
            if ctrl_right is not None:
                state_r = ctrl_right.get_state()
                eef_r = state_r["eef"] if state_r["eef"] is not None else np.zeros(6)
                grip_r = state_r["qpos"][-1]

            proprio = build_proprio(eef_l, qpos_l[-1], right_eef_xyzrpy=eef_r, right_gripper=grip_r)
            obs = {
                "proprio": proprio,
                "images": frames,
                "language": args.language,
                "domain_id": SERVER["domain_id"],
            }
            raw_action = client.predict(obs)
            split = split_action(raw_action)

            if args.mode == "eef" and "left_eef" in split:
                left = split["left_eef"]
                ctrl_left.move_pose(np.array(left[:6]), gripper=float(left[6]))
                if ctrl_right and "right_eef" in split:
                    right = split["right_eef"]
                    ctrl_right.move_pose(np.array(right[:6]), gripper=float(right[6]))
            elif "left_joint" in split:
                ctrl_left.move_joint(np.array(split["left_joint"][:7]))
                if ctrl_right and "right_joint" in split:
                    ctrl_right.move_joint(np.array(split["right_joint"][:7]))
            else:
                raise RuntimeError("Control mode/action mismatch.")

            elapsed = time.time() - t0
            sleep_time = max(0.0, rate - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cam_mgr.release_all()
        ctrl_left.disable()
        if ctrl_right:
            ctrl_right.disable()


if __name__ == "__main__":
    main()
