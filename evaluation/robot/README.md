# ROS-free SoftFold-Agilex 说明

## 架构概览
- 目录结构：`camera/` (OpenCV / RealSense 采集)、`piper/` (Piper 控制 + IK)、`rotation.py` (6D↔Euler 转换)、`run_client.py` (抓图→HTTP 推理→下发)。
- 依赖：`opencv-python`, `requests`, `pinocchio`, `casadi`, `piper_sdk`（供应商 SDK），Python 3.10 推荐；若 `cam_high` 用 RealSense/D435i，需安装 `pyrealsense2`。
- 观测输入：多路相机帧 + 末端 6D proprio（xyz + rot6d + gripper，左右各 10，缺失时补零）。
- 动作输出：默认期望 20 维 eef_6d（左右各 10），经 IK → 关节 → CAN；若模型输出 14 维关节，可用 `--mode joint` 直接下发。

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
- 控制与单位：默认走 **末端 6D + IK 下发**（与 SoftFold-Agilex 数据的 eef_6d 对齐），模型输出弧度/米，内部转换为 0.001°/1e6 夹爪。`PIPER` 字段可调关节/夹爪缩放、回零/安全位。若模型直接输出关节，可用 `--mode joint` 兼容。
- IK：基于 Pinocchio+CasADi 的简单 IK（`piper/kinematics.py`），用于将 6D 末端转换为关节后写 CAN。
- 开机测试：
  - 点动示例：`python evaluation/robot/piper/demo_jog.py --can can0`
  - 命令说明：自动 Enable → 回零 → 移动到简单姿态 → Disable。

## 无 ROS 运行推理
1) 启动 X-VLA 推理服务（参考根目录 `deploy.py`）。
2) 运行客户端（默认末端 6D → IK）：`python evaluation/robot/run_client.py --host <srv_ip> --port 8000 --chunk_size 10`
   - 默认使用三路相机 `cam_high/cam_left_wrist/cam_right_wrist`，以及左臂 CAN。
   - 将实时帧与当前末端/关节状态打包成 HTTP 请求 `/act`，解析动作后默认 **6D 末端→IK→关节下发**（长度 20）。若动作长度 14（关节），需加 `--mode joint`。
   - 6D↔Euler 转换使用本地 `evaluation/robot/rotation.py`，避免与上层 `deploy.py` 同名模块冲突。

## 常见排查
- 相机打开失败：检查 `config.py` 的 `device_index/serial_hint` 与实际 `/dev/videoX` 序列号是否一致。
- 帧率/分辨率无效：某些驱动不支持所填值，降低分辨率或用 `v4l2-ctl --list-formats-ext` 查看支持的模式。
- CAN 未连接：确认 `ip link set can0 up type can bitrate 1000000` 已配置，换成你的实际端口名；`piper_sdk` 能正常连接。 

## 快速测试路径
1) **相机预览**：`python evaluation/robot/camera/preview.py --duration 10`，确认三路画面正常。
2) **IK 离线验证（可选）**：`python - <<'PY' ...` 读入 `observations/eef_6d`，调用 `PiperIKSolver.solve_pose` 确认可解。
3) **端到端推理**：确保 `piper_sdk` 安装、CAN 接好，运行 `python evaluation/robot/run_client.py --host <srv_ip> --port 8000 --chunk_size 10`（默认 eef）。模型若输出关节，则加 `--mode joint`。
4) **安全位检查**：`config.PIPER.home_rad/safe_rad` 设为安全姿态，上电/断电时会使用。
