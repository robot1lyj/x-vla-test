import logging
import numpy as np
from typing import Optional

RAD_TO_MDEG = 57324.840764  # SoftFold/ROS node factor (rad -> milli-degree)
MDEG_TO_RAD = 1.0 / RAD_TO_MDEG


def _instantiate_interface(can_name: str):
    """Try Piper SDK constructors in order."""

    try:
        from piper_sdk import C_PiperInterface_V2  # type: ignore

        try:
            return C_PiperInterface_V2(can_name=can_name)
        except TypeError:
            return C_PiperInterface_V2(can_name)
    except ImportError:
        pass

    try:
        from piper_sdk import C_PiperInterface  # type: ignore

        try:
            return C_PiperInterface(can_name=can_name)
        except TypeError:
            return C_PiperInterface(can_name)
    except ImportError as exc:
        raise ImportError("piper_sdk is required to talk to the Piper arm") from exc


class PiperBus:
    """Lightweight wrapper around piper_sdk with SoftFold-compatible units."""

    def __init__(
        self,
        can_name: str,
        joint_factor: float = RAD_TO_MDEG,
        gripper_scale: float = 1_000_000.0,
        gripper_max: int = 80_000,
        motion_speed: int = 100,
    ) -> None:
        self.can_name = can_name
        self.joint_factor = float(joint_factor)
        self.gripper_scale = float(gripper_scale)
        self.gripper_max = int(gripper_max)
        self.motion_speed = int(motion_speed)
        self.piper = _instantiate_interface(can_name)
        self.piper.ConnectPort()
        self.enabled = False

    def enable(self) -> None:
        if self.enabled:
            return
        self.piper.EnableArm(7)
        # Default to joint mode and zero gripper.
        self._motion_mode()
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        self.enabled = True
        logging.info("Piper enabled on %s", self.can_name)

    def disable(self) -> None:
        if not self.enabled:
            return
        self.piper.DisableArm(7)
        self.enabled = False
        logging.info("Piper disabled on %s", self.can_name)

    def _motion_mode(self) -> None:
        """Best-effort MotionCtrl_2 variant (API differs across SDK versions)."""

        try:
            self.piper.MotionCtrl_2(0x01, 0x01, self.motion_speed, 0xAD)
        except TypeError:
            self.piper.MotionCtrl_2(0x01, 0x01, self.motion_speed)

    def write(self, joints_rad: np.ndarray, gripper: float = 0.0) -> None:
        if not self.enabled:
            raise RuntimeError("Arm not enabled. Call enable() first.")
        if joints_rad.shape[0] < 6:
            raise ValueError("Need at least 6 joint targets")

        cmd = [int(round(v * self.joint_factor)) for v in joints_rad[:6]]
        grip_dev = int(round(gripper * self.gripper_scale))
        grip_dev = max(0, min(self.gripper_max, grip_dev))

        self._motion_mode()
        self.piper.JointCtrl(*cmd)
        self.piper.GripperCtrl(grip_dev, 1000, 0x01, 0)
        self._motion_mode()

    def read_joints(self) -> np.ndarray:
        msg = self.piper.GetArmJointMsgs().joint_state
        joints = np.array(
            [
                msg.joint_1 * MDEG_TO_RAD,
                msg.joint_2 * MDEG_TO_RAD,
                msg.joint_3 * MDEG_TO_RAD,
                msg.joint_4 * MDEG_TO_RAD,
                msg.joint_5 * MDEG_TO_RAD,
                msg.joint_6 * MDEG_TO_RAD,
            ],
            dtype=float,
        )
        grip_msg = self.piper.GetArmGripperMsgs().gripper_state
        gripper = grip_msg.grippers_angle / self.gripper_scale
        return np.concatenate([joints, np.array([gripper])], axis=0)

    def read_eef(self) -> Optional[np.ndarray]:
        if not hasattr(self.piper, "GetArmEndPoseMsgs"):
            return None
        end = self.piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([end.X_axis, end.Y_axis, end.Z_axis], dtype=float) / 1_000_000.0
        rpy_deg = np.array([end.RX_axis, end.RY_axis, end.RZ_axis], dtype=float) / 1000.0
        rpy = np.deg2rad(rpy_deg)
        return np.concatenate([pos, rpy], axis=0)


__all__ = ["PiperBus", "RAD_TO_MDEG", "MDEG_TO_RAD"]
