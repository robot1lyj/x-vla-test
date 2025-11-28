import logging
from typing import Dict

from evaluation.robot.camera.opencv_camera import OpenCVCamera, discover_cameras
from evaluation.robot.camera.realsense_camera import RealSenseCamera


class CameraManager:
    def __init__(self, cam_cfg: Dict[str, Dict]):
        """
        cam_cfg: {name: {driver, serial_hint, device_index, width, height, fps}}
        """
        self.cam_cfg = cam_cfg
        self.handles: Dict[str, object] = {}
        self.index_map: Dict[str, int] = {}

    def resolve_indices(self) -> None:
        serial_map = discover_cameras()
        for name, cfg in self.cam_cfg.items():
            driver = cfg.get("driver", "opencv")
            if driver == "realsense":
                self.index_map[name] = -1
                continue
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
            driver = cfg.get("driver", "opencv")
            if driver == "realsense":
                handle = RealSenseCamera(cfg.get("serial_hint", ""), cfg["width"], cfg["height"], cfg["fps"])
            else:
                idx = self.index_map[name]
                handle = OpenCVCamera(idx, cfg["width"], cfg["height"], cfg["fps"])
            if not handle.open():
                raise RuntimeError(f"Failed to open camera {name}")
            self.handles[name] = handle
            logging.info("Opened camera %s with driver %s", name, driver)

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
            if hasattr(handle, "release"):
                handle.release()
        self.handles.clear()
