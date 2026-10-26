# Multi-Robot Deployment Code

Python scripts accompanying Meghana Chintalapati's thesis, *Multi-Robot Foraging in Structured Environments Using MAPPO and Token-Based Incentives*.

The collection covers overhead robot localization, closed-loop wheel control, robotic-arm routine integration, offline object detection, and ROS 2/RViz policy-execution components.

## File Guide

| File | Purpose |
| --- | --- |
| `hardware/aruco_multi_robot_odom_robust3.py` | RealSense-based ArUco tracking, robot identification, pose filtering, and ROS 2 odometry publication. |
| `hardware/topic_combined_final2.py` | Raspberry Pi wheel control using position and heading feedback, with an interface to the arm-collection routine. |
| `hardware/variants/topic_combined_final.py` | Alternative wheel-controller implementation. |
| `perception/yolo_test.py` | YOLO detection and annotated video generation from recorded footage. |
| `ros_rviz/new_dynamic.py` | Multi-robot TF simulation and waypoint execution for RViz. |
| `ros_rviz/board_return.py` | Grid-state management, capability-based collection checks, task events, and waypoint publication. |
| `ros_rviz/lbforaging.py` | Base 21-feature LBF observation construction and publication. |
| `ros_rviz/rlloop.py` | GRU actor inference and coordination of actions with execution acknowledgments. |
| `ros_rviz/action_replay.py` | Action-sequence replay for execution testing. |
| `ros_rviz/lidar_os.py` | Local field-of-view and obstacle-state visualization from board and TF information. |
| `ros_rviz/lidar_data.py` | Local field-of-view diagnostics from board and TF information. |

## Dependencies

- Python 3 and ROS 2 with `rclpy`.
- ROS 2 message packages: `std_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, and `visualization_msgs`.
- `tf2_ros`, `cv_bridge`, and `robot_state_publisher` for the relevant ROS components.
- NumPy and PyTorch for observation processing and policy inference.
- OpenCV with ArUco support and `pyrealsense2` for overhead localization.
- Ultralytics for YOLO detection.
- `RPi.GPIO` on the Raspberry Pi for motor control.

Source the ROS 2 environment in each terminal and install the dependencies required by the selected script.

## Configuration

Before execution, set the model and input-video paths, robot descriptions, grid configuration, marker IDs, ROS topics, and arm-routine path for the selected setup.

The hardware scripts use robot0–robot2, while the RViz scripts use robot1–robot3. Match robot names and coordinate frames when connecting components. The base observation builder publishes 21 features; configure observation range, capability levels, obstacle encoding, and actor structure to match the selected checkpoint.

Physical positions are expressed in normalized image coordinates. Target points and odometry must use the same coordinate convention. The controller receives target positions on `/target_points` as `PoseArray`; pose x/y hold coordinates and z identifies the target point.

## Hardware Interfaces

| Component | Input | Output |
| --- | --- | --- |
| ArUco localization | RealSense RGB images | `/robot0/odom`, `/robot1/odom`, `/robot2/odom`, `/topview/raw`, `/topview/labeled` |
| robot0 motion controller | `/robot0/odom`, `/target_points`, `/robot0/action` | Wheel direction/PWM signals and arm-routine invocation |

GPIO assignments use BCM numbering:

| Wheel | Direction GPIOs | PWM GPIO |
| --- | --- | --- |
| Front left | 17, 27 | 18 |
| Front right | 22, 23 | 19 |
| Rear left | 5, 6 | 12 |
| Rear right | 13, 26 | 16 |

PWM frequency is 100 Hz. The controller includes a 0.02-s loop sleep and position stopping tolerances of 0.010 in normalized coordinates. The `final2` controller applies a 60-s timeout to the external arm-collection routine.

## Running the Scripts

Run commands from this folder after configuring each component's inputs.

### Overhead localization

The supplied marker mapping uses IDs 0–3 for robot0, 4–7 for robot1, and 8–11 for robot2. Initially expose three or four markers per robot to initialize the stored geometry.

```bash
python3 hardware/aruco_multi_robot_odom_robust3.py
```

### Physical motion control

On the Raspberry Pi, start the configured target-point and action publishers, then run one matching wheel controller:

```bash
python3 hardware/topic_combined_final2.py
```

The arm-collection interface invokes `sub_routine_catch2.py`; set its path to the routine used on the robot. Configure robot-specific topic names for each platform.

### Recorded-video detection

Set `VIDEO_PATH`, `MODEL_PATH`, and `OUTPUT_PATH` in the script:

```bash
python3 perception/yolo_test.py
```

### ROS 2/RViz execution

Start components in this order, configuring the matching URDF assets, checkpoint, and observation representation:

1. Waypoint simulator: `new_dynamic.py`.
2. Grid/task server: `board_return.py`.
3. Observation construction and visualization nodes.
4. Policy inference through `rlloop.py`, or scripted testing through `action_replay.py`.
5. RViz with TF and marker displays.

For example:

```bash
python3 ros_rviz/new_dynamic.py
python3 ros_rviz/board_return.py
```

Run each node in a separate terminal. Use one action source at a time. Configure `URDF_DIR` for the simulator and `checkpoint_path` for policy inference.

## Related Learning Framework

The MAPPO training, evaluation, and incentive-analysis framework is maintained separately:

https://github.com/sanjayadpf/MOMARL_RGPL_MAPPO

## Source Record

`SOURCE_MANIFEST.json` records the supplied filenames and SHA-256 checksums. The Python source files are preserved as supplied. Retain original author comments and acknowledge collaborators and upstream libraries when reusing the code.
