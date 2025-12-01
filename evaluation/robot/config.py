"""
Base configuration for the ROS-free SoftFold-Agilex runner.

Adjust device indices/serials, URDF path, and inference server endpoints here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Camera mapping: fill either serial substring or device index.
# driver: "opencv" (VideoCapture) or "realsense" (pyrealsense2). cam_high 默认 RealSense D435i。
CAMERAS = {
    "cam_high": {"driver": "realsense", "serial_hint": "", "device_index": -1, "width": 640, "height": 480, "fps": 30},
    "cam_left_wrist": {"driver": "opencv", "serial_hint": "", "device_index": 8, "width": 640, "height": 480, "fps": 30},
    "cam_right_wrist": {"driver": "opencv", "serial_hint": "", "device_index": 6, "width": 640, "height": 480, "fps": 30},
}

# CAN bus ports for dual arms. If你的系统网络接口名就是 can_left/can_right，直接填同名。
# 示例: {"can_left": "can_left", "can_right": "can_right"}
CANS = {"can_left": "can_left", "can_right": "can_right"}

# Mechanical arm settings.
URDF_PATH = ROOT / "SoftFold-Agilex" / "Piper_ros_private-ros-noetic" / "src" / "piper_description" / "urdf" / "piper_description.urdf"

# IK solver options (eef control).
IK = {
    "max_iter": 50,
    "tol": 1e-4,
    "w_pose": 20.0,
    "w_reg": 0.01,
    "smooth_weight": 0.0,
    "jump_threshold_rad": 30.0 / 180.0 * 3.1415926,
    "trust_region": 0.35,
}

# Control mode: "eef" (default, 6D action + IK) or "joint".
CONTROL_MODE = "eef"

# Piper CAN / unit settings (SoftFold-compatible).
PIPER = {
    "joint_factor": 57324.840764,  # rad -> 0.001 deg (SoftFold SDK)
    "gripper_scale": 1_000_000.0,  # gripper unit scaling (meters -> device)
    "gripper_max": 80_000,
    "motion_speed": 100,
    "home_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "safe_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Inference server.
SERVER = {"host": "0.0.0.0", "port": 8000, "domain_id": 5, "chunk_size": 10}

# Motion constraints.
MOTION = {
    "publish_rate": 15,
    "joint_limits_margin": 0.0,
    "max_step_rad": 10.0 / 180.0 * 3.1415926,
    "wrist_weight": 1.0,
}
