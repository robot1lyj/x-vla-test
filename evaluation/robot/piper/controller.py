import logging
import math
from typing import Dict, Optional

import numpy as np

from evaluation.robot.config import CONTROL_MODE, IK, MOTION, PIPER, URDF_PATH
from evaluation.robot.piper.bus import PiperBus
from evaluation.robot.piper.kinematics import PiperIKSolver
from evaluation.robot.piper.safety import limit_velocity


class PiperController:
    """Piper controller with direct joint commands (SoftFold default)."""

    def __init__(self, can_port: str = "can0"):
        bus_cfg = PIPER
        self.bus = PiperBus(
            can_name=can_port,
            joint_factor=bus_cfg.get("joint_factor", 180.0 / math.pi * 1000.0),
            gripper_scale=bus_cfg.get("gripper_scale", 1_000_000.0),
            gripper_max=bus_cfg.get("gripper_max", 80_000),
            motion_speed=bus_cfg.get("motion_speed", 100),
        )
        self.enabled = False
        self.home_q = np.asarray(bus_cfg.get("home_rad", [0, 0, 0, 0, 0, 0, 0]), dtype=float)
        self.safe_q = np.asarray(bus_cfg.get("safe_rad", self.home_q), dtype=float)

        self.ik: Optional[PiperIKSolver] = None
        if CONTROL_MODE == "eef":
            self.ik = PiperIKSolver(
                str(URDF_PATH),
                weight_pose=IK["w_pose"],
                weight_reg=IK["w_reg"],
                smooth_weight=IK.get("smooth_weight", 0.0),
                max_iter=IK["max_iter"],
                tol=IK["tol"],
                jump_threshold_rad=IK["jump_threshold_rad"],
                trust_region=IK.get("trust_region"),
            )
        self.last_q: Optional[np.ndarray] = None

    # ---- Low-level helpers ----
    def enable(self):
        if self.enabled:
            return
        self.bus.enable()
        self.enabled = True

    def disable(self):
        if self.enabled:
            try:
                if self.safe_q is not None and self.safe_q.size >= 6:
                    self.move_joint(self.safe_q, gripper=float(self.safe_q[-1]) if self.safe_q.size > 6 else 0.0)
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Failed to move to safe pose before disable: %s", exc)
            self.bus.disable()
            self.enabled = False

    # ---- State ----
    def get_joint_state(self) -> np.ndarray:
        return self.bus.read_joints()

    def get_eef_pose(self) -> np.ndarray:
        eef = self.bus.read_eef()
        if eef is not None:
            return eef
        if self.ik is not None:
            q = self.get_joint_state()[: self.ik.reduced_robot.model.nq]
            return self.ik.forward_k(q)
        return None

    # ---- Commands ----
    def move_joint(self, joints_rad: np.ndarray, gripper: Optional[float] = None):
        if not self.enabled:
            raise RuntimeError("Arm not enabled. Call enable() first.")
        if self.last_q is not None:
            joints_rad, clipped = limit_velocity(
                self.last_q,
                joints_rad,
                max_step=MOTION.get("max_step_rad", math.radians(10)),
            )
            if clipped:
                logging.debug("Joint step limited.")

        grip = gripper
        if grip is None and joints_rad.shape[0] > 6:
            grip = float(joints_rad[6])
        if grip is None:
            grip = 0.0

        self.bus.write(joints_rad, gripper=grip)
        self.last_q = joints_rad

    def home(self):
        self.move_joint(self.home_q, gripper=float(self.home_q[-1]) if self.home_q.size > 6 else None)

    def get_state(self) -> Dict[str, np.ndarray]:
        q = self.get_joint_state()
        eef = self.get_eef_pose()
        return {"qpos": q, "eef": eef}


__all__ = ["PiperController"]
