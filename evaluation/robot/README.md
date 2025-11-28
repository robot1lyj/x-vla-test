# ROS-free SoftFold-Agilex 说明

- 目录结构：`camera/` (OpenCV / RealSense 采集)、`piper/` (Piper 控制 + IK)、`run_client.py` (抓图→HTTP 推理→下发)。
- 依赖：`opencv-python`, `requests`, `pinocchio`, `casadi`, `piper_sdk`（供应商 SDK），Python 3.10 推荐；若 `cam_high` 用 RealSense/D435i，需安装 `pyrealsense2`。

## 相机
- 驱动选择：`config.py` 的 `CAMERAS` 每路可设 `driver`，`opencv` 走 `/dev/video*`，`realsense` 走 `pyrealsense2`（如 D435i）。
- 串口号/序列号获取：
  - OpenCV 相机：`ls /dev/video*`，`cat /sys/class/video4linux/videoX/device/serial`，或 `udevadm info /dev/videoX | grep ID_SERIAL_SHORT`
  - RealSense：`rs-enumerate-devices` 查看序列号（若只有一台可留空自动匹配）。
- 交互式识别（推荐）：`python evaluation/robot/camera/identify.py` 按提示拔插 `cam_high/cam_left_wrist/cam_right_wrist`，自动输出 `serial_hint/device_index` 建议值（仅针对 OpenCV 相机；RealSense 请填序列号或留空）。
- 配置分辨率/帧率：修改 `evaluation/robot/config.py` 中 `CAMERAS`，设置 `driver/serial_hint/device_index/width/height/fps`。
- 预览测试：`python evaluation/robot/camera/preview.py --duration 10`，按 `q` 退出；拼接显示三路视角。

## 机械臂（Piper）
- CAN 口设置：`config.py` 中 `CANS` 填你实际的接口名。如果系统里网卡名就是 `can_left/can_right`，可直接写 `{"can_left": "can_left", "can_right": "can_right"}`。
- 开机测试：
  - 点动示例：`python evaluation/robot/piper/demo_jog.py --can can0`
  - 命令说明：自动 Enable → 回零 → IK 移动到简单姿态 → Disable。
- IK 参数：`config.py` 的 `IK` 字段可调 `max_iter/tol`、跳变阈值等；关节限幅 margin、发布频率在 `MOTION` 中。

## 无 ROS 运行推理
1) 启动 X-VLA 推理服务（参考根目录 `deploy.py`）。
2) 运行客户端：`python evaluation/robot/run_client.py --host <srv_ip> --port 8000 --chunk_size 10`
   - 默认使用三路相机 `cam_high/cam_left_wrist/cam_right_wrist`，以及左臂 CAN。
   - 将实时帧与当前末端状态打包成 HTTP 请求 `/act`，解析动作后做 IK 下发。

## 常见排查
- 相机打开失败：检查 `config.py` 的 `device_index/serial_hint` 与实际 `/dev/videoX` 序列号是否一致。
- 帧率/分辨率无效：某些驱动不支持所填值，降低分辨率或用 `v4l2-ctl --list-formats-ext` 查看支持的模式。
- CAN 未连接：确认 `ip link set can0 up type can bitrate 1000000` 已配置，换成你的实际端口名；`piper_sdk` 能正常连接。 
