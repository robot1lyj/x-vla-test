import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure repo root on sys.path for evaluation.* imports.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.robot.camera.camera_manager import discover_cameras
from evaluation.robot.config import CAMERAS


def wait_for_change(prev: Dict[int, str], timeout: float = 30.0, interval: float = 0.5) -> Tuple[Dict[int, str], Dict[int, str]]:
    start = time.time()
    while time.time() - start < timeout:
        cur = discover_cameras()
        if cur != prev:
            return cur, cur
        time.sleep(interval)
    raise TimeoutError("未检测到新设备变化，请重试。")


def pick_new_device(prev: Dict[int, str], cur: Dict[int, str]) -> Tuple[int, str]:
    added = [idx for idx in cur.keys() if idx not in prev]
    if added:
        idx = added[0]
        return idx, cur[idx]
    changed = [idx for idx in cur.keys() if idx in prev and cur[idx] != prev[idx]]
    if changed:
        idx = changed[0]
        return idx, cur[idx]
    raise RuntimeError("检测到设备变化但无法定位新设备。")


def main():
    parser = argparse.ArgumentParser(description="提示插拔相机并自动记录串口号/设备号。按顺序识别多路相机。")
    parser.add_argument(
        "--names",
        nargs="+",
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="要识别的相机逻辑名称（与 config.py 的 CAMERAS 键对应）。",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="等待插拔的超时时间（秒）。")
    args = parser.parse_args()

    baseline = discover_cameras()
    print("当前已检测到的设备 (index -> serial):", baseline)

    results = {}
    prev = baseline
    for name in args.names:
        input(f"\n准备识别 {name}。请拔掉并重新插上该相机，插好后按回车继续检测...")
        try:
            cur, _ = wait_for_change(prev, timeout=args.timeout)
            idx, serial = pick_new_device(prev, cur)
            results[name] = {"device_index": idx, "serial": serial}
            prev = cur
            print(f"检测到 {name}: /dev/video{idx}, serial='{serial}'")
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] 识别失败: {exc}")
            return

    print("\n识别完成，可将以下内容填入 evaluation/robot/config.py 的 CAMERAS 对应项：")
    for name, info in results.items():
        print(
            f"{name}: serial_hint='{info['serial']}', device_index={info['device_index']}, width=640, height=480, fps=30"
        )


if __name__ == "__main__":
    main()
