import argparse
import glob
import sys
import time
from pathlib import Path

# Ensure repo root on sys.path for evaluation.* imports.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.robot.config import CAMERAS


def list_video_ports():
    """List /dev/video* paths."""
    return sorted(glob.glob("/dev/video*"))


def main():
    parser = argparse.ArgumentParser(description="提示拔插相机，自动记录 /dev/videoX 对应的逻辑名称（OpenCV 相机）。")
    parser.add_argument(
        "--names",
        nargs="+",
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="要识别的相机逻辑名称（与 config.py 的 CAMERAS 键对应）。",
    )
    args = parser.parse_args()

    print("当前已检测到的设备:", list_video_ports())

    results = {}
    for name in args.names:
        driver = CAMERAS.get(name, {}).get("driver", "opencv")
        if driver != "opencv":
            print(f"\n{name} 设置为 driver={driver}，跳过拔插检测（RealSense 请用序列号或留空自动匹配）。")
            continue

        ports_before = list_video_ports()
        print(f"\n准备识别 {name} (OpenCV)。请拔掉该相机 USB，再按回车继续。")
        input()
        time.sleep(0.5)
        ports_after_unplug = list_video_ports()
        removed = list(set(ports_before) - set(ports_after_unplug))

        print("现在请重新插上该相机 USB，再按回车继续。")
        input()
        time.sleep(0.5)
        ports_after_replug = list_video_ports()
        added = list(set(ports_after_replug) - set(ports_after_unplug))

        if len(removed) == 1 and len(added) == 1:
            dev = added[0]
            idx = int(dev.replace("/dev/video", ""))
            results[name] = {"device_index": idx, "serial": ""}
            print(f"检测到 {name}: {dev}")
        else:
            print(f"[{name}] 识别失败，检测到删除 {removed}，新增 {added}，请重试。")

    if results:
        print("\n识别完成，可将以下内容填入 evaluation/robot/config.py 的 CAMERAS 对应项：")
        for name, info in results.items():
            print(f"{name}: serial_hint='{info['serial']}', device_index={info['device_index']}, width=640, height=480, fps=30")


if __name__ == "__main__":
    main()
