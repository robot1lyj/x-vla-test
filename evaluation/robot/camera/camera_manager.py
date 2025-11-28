import glob
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2


def _read_serial(dev_path: Path) -> str:
    """Try to read camera serial from sysfs."""
    # Example: /sys/class/video4linux/video0/device/serial
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


class CameraHandle:
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


class CameraManager:
    def __init__(self, cam_cfg: Dict[str, Dict]):
        """
        cam_cfg: {name: {serial_hint: str, device_index: int, width, height, fps}}
        """
        self.cam_cfg = cam_cfg
        self.handles: Dict[str, CameraHandle] = {}
        self.index_map: Dict[str, int] = {}

    def resolve_indices(self) -> None:
        serial_map = discover_cameras()
        for name, cfg in self.cam_cfg.items():
            serial_hint = cfg.get("serial_hint") or ""
            preferred = cfg.get("device_index", -1)
            matched_idx = None
            if serial_hint:
                for idx, serial in serial_map.items():
                    if serial_hint in serial:
                        matched_idx = idx
                        break
            if matched_idx is None and preferred >= 0:
                matched_idx = preferred
            if matched_idx is None:
                raise RuntimeError(f"Camera {name} not found via serial or device_index.")
            self.index_map[name] = matched_idx

    def open_all(self) -> None:
        self.resolve_indices()
        for name, cfg in self.cam_cfg.items():
            idx = self.index_map[name]
            handle = CameraHandle(idx, cfg["width"], cfg["height"], cfg["fps"])
            if not handle.open():
                raise RuntimeError(f"Failed to open camera {name} at /dev/video{idx}")
            self.handles[name] = handle
            logging.info("Opened camera %s on /dev/video%d", name, idx)

    def read_all(self) -> Dict[str, any]:
        frames = {}
        for name, handle in self.handles.items():
            ok_frame = handle.read()
            if ok_frame is None:
                raise RuntimeError(f"Camera {name} not opened")
            ok, frame = ok_frame
            if not ok:
                raise RuntimeError(f"Camera {name} failed to grab frame")
            frames[name] = frame
        return frames

    def release_all(self) -> None:
        for handle in self.handles.values():
            handle.release()
        self.handles.clear()

