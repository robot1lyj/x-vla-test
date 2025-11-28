import logging
import math
from typing import Dict, Optional

import numpy as np

from evaluation.robot.config import IK, MOTION, URDF_PATH
from evaluation.robot.piper.kinematics import ArmIK, xyzrpy_to_se3
from evaluation.robot.piper.safety import clamp_to_limits, limit_velocity

try:
    from piper_sdk import C_PiperInterface
except ImportError as exc:  # pragma: no cover - hardware dependency
    raise ImportError("piper_sdk is required to control the Piper arm.") from exc


class PiperController:
    def __init__(self, can_port: str = "can0"):
        self.can_port = can_port
        self.piper = C_PiperInterface(can_name=can_port)
        self.piper.ConnectPort()
        self.enabled = False

        self.ik = ArmIK(
            str(URDF_PATH),
            weight_pose=IK["w_pose"],
            weight_reg=IK["w_reg"],
            smooth_weight=IK["smooth_weight"],
            max_iter=IK["max_iter"],
            tol=IK["tol"],
            jump_threshold_rad=IK["jump_threshold_rad"],
        )
        self.last_q: Optional[np.ndarray] = None

    # ---- Low-level helpers ----
    @staticmethod
    def _rad_to_device(rad: float) -> int:
        factor = 57324.840764  # 1000 * 180 / pi
        return int(round(rad * factor))

    @staticmethod
    def _grip_to_device(grip: float) -> int:
        val = int(round(grip * 1_000_000))
        return max(0, min(80000, val))

    def enable(self):
        if self.enabled:
            return
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.enabled = True
        logging.info("Piper enabled on %s", self.can_port)

    def disable(self):
        self.piper.DisableArm(7)
        self.enabled = False
        logging.info("Piper disabled on %s", self.can_port)

    # ---- State ----
    def get_joint_state(self) -> np.ndarray:
        msg = self.piper.GetArmJointMsgs().joint_state
        factor = 0.017444  # radians per milli-degree
        joints = np.array(
            [
                msg.joint_1 / 1000 * factor,
                msg.joint_2 / 1000 * factor,
                msg.joint_3 / 1000 * factor,
                msg.joint_4 / 1000 * factor,
                msg.joint_5 / 1000 * factor,
                msg.joint_6 / 1000 * factor,
                self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1_000_000,
            ],
            dtype=float,
        )
        return joints

    def get_eef_pose(self) -> np.ndarray:
        end = self.piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([end.X_axis, end.Y_axis, end.Z_axis], dtype=float) / 1_000_000.0
        rpy = np.array([end.RX_axis, end.RY_axis, end.RZ_axis], dtype=float) / 1000.0
        rpy = np.deg2rad(rpy)
        return np.concatenate([pos, rpy], axis=0)

    # ---- Commands ----
    def move_joint(self, joints_rad: np.ndarray, gripper: float = 0.0):
        if not self.enabled:
            raise RuntimeError("Arm not enabled. Call enable() first.")
        joints_rad = clamp_to_limits(
            joints_rad,
            self.ik.reduced_robot.model.lowerPositionLimit,
            self.ik.reduced_robot.model.upperPositionLimit,
            margin=MOTION["joint_limits_margin"],
        )
        if self.last_q is not None:
            joints_rad, clipped = limit_velocity(self.last_q, joints_rad, max_step=math.radians(10))
            if clipped:
                logging.debug("Joint step limited.")
        dev = [self._rad_to_device(v) for v in joints_rad[:6]]
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(*dev)
        self.piper.GripperCtrl(self._grip_to_device(gripper), 1000, 0x01, 0)
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.last_q = joints_rad

    def move_pose(self, xyzrpy: np.ndarray, gripper: float = 0.0):
        target = xyzrpy_to_se3(xyzrpy)
        q_init = self.last_q
        q, success, collision = self.ik.solve(target, q_init=q_init)
        if not success:
            raise RuntimeError("IK failed.")
        if collision:
            logging.warning("IK solution in collision.")
        self.move_joint(q, gripper=gripper)

    def home(self):
        zeros = np.zeros(7)
        self.move_joint(zeros)

    def get_state(self) -> Dict[str, np.ndarray]:
        q = self.get_joint_state()
        eef = self.ik.forward_k(q[: self.ik.reduced_robot.model.nq])
        return {"qpos": q, "eef": eef}
