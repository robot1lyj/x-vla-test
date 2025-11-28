import glob
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2


def _read_serial(dev_path: Path) -> str:
    """Try to read camera serial from sysfs."""
    serial_file = dev_path / "device" / "serial"
    if serial_file.exists():
        try:
            return serial_file.read_text().strip()
        except OSError:
            return ""
    return ""


def discover_cameras() -> Dict[int, str]:
    """Return mapping: device index -> serial (if found)."""
    mapping: Dict[int, str] = {}
    for dev in glob.glob("/dev/video*"):
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue
        serial = _read_serial(Path(f"/sys/class/video4linux/video{idx}"))
        mapping[idx] = serial
    return mapping


class OpenCVCamera:
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            logging.error("Open camera %s failed", self.index)
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read(self) -> Optional[Tuple[bool, any]]:
        if self.cap is None:
            return None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

