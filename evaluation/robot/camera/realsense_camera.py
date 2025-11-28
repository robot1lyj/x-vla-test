import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:
    rs = None


def discover_realsense() -> Dict[str, str]:
    """Return mapping: serial -> name (if pyrealsense2 available)."""
    if rs is None:
        return {}
    ctx = rs.context()
    devices = ctx.query_devices()
    return {dev.get_info(rs.camera_info.serial_number): dev.get_info(rs.camera_info.name) for dev in devices}


class RealSenseCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        if rs is None:
            raise ImportError("pyrealsense2 is required for RealSense driver.")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None

    def open(self) -> bool:
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config = cfg
        try:
            self.pipeline.start(cfg)
            return True
        except Exception as exc:  # noqa: BLE001
            logging.error("Open RealSense %s failed: %s", self.serial or "any", exc)
            return False

    def read(self) -> Optional[Tuple[bool, any]]:
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        frame = np.asanyarray(color_frame.get_data())
        return True, frame

    def release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

