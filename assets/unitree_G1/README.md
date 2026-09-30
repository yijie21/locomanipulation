# Ether Robot Assets

This repository contains robot configuration files, URDF models, and calibration data for Ether robotics projects. Each branch represents a specific robot instance with its own hardware configuration and sensor setup.

## Overview

- **Current Branch**: `saturn` - Unitree G1-EDU-U2 robot instance
- **Robot Model**: G1-EDU-U2 (Unitree)
- **Serial Number**: E21D6000Q5AAKA03
- **Manufacture Date**: 2026-05-10

## Project Structure

```
ether-robot-assets/
├── urdf/                          # Robot URDF models
│   ├── g1_comp_dex1_1_backpack.urdf   # Main robot URDF file
│   └── meshes/                    # 3D mesh files for visualization
├── calibration/                   # Robot calibration data
├── robot.yaml                     # Robot configuration and exports
├── view_urdf.py                   # Interactive URDF viewer script
└── pyproject.toml                 # Python project configuration
```

## Robot Configuration

The `robot.yaml` file defines:

### Exports
- `ETHER_ROBOT_URDF`: Path to the main URDF file
- `ETHER_ROBOT_CALIB_DIR`: Directory containing calibration data
- `ETHER_LIVOX_CONFIG`: Lidar configuration file (optional)

### Runtime Configuration
- Robot identity: Make, model, and serial numbers
- Sensors:
  - **D435 (Head)**: RGB-D camera on Orin computer
  - **D455 (Head)**: RGB-D camera on Thor computer
  - **D405 (Left Wrist)**: RGB-D camera on Thor computer
  - **D405 (Right Wrist)**: RGB-D camera on Thor computer

## Setup

### Prerequisites
- Python 3.10+
- MuJoCo physics engine

### Installation

Using `uv` (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install mujoco>=3.2.0
```

## Usage

### Interactive URDF Viewer

Interactively inspect and test the robot's URDF with slider controls:

```bash
# Load the default URDF from robot.yaml
python view_urdf.py

# Or load a specific URDF
python view_urdf.py path/to/robot.urdf
```

**Features:**
- Visual 3D model of the robot
- Interactive joint sliders in the "Control" tab
- Each slider corresponds to a joint and is bounded by URDF limits
- Gravity disabled for easy inspection
- Test joint ranges, mesh collisions, and gripper travel

**Controls in Viewer:**
- Right-hand panel → "Control" tab to access joint sliders
- Drag sliders to move joints through their full range of motion

### Python API

Load and inspect the URDF programmatically:

```python
import mujoco
import yaml

# Load robot configuration
with open("robot.yaml", "r") as f:
    config = yaml.safe_load(f)
    urdf_path = config["exports"]["ETHER_ROBOT_URDF"]

# Load URDF with MuJoCo
model = mujoco.MjSpec.from_string(urdf_text).compile()
data = mujoco.MjData(model)

# Access robot properties
print(f"Degrees of Freedom: {model.nq}")
print(f"Actuated Joints: {model.nu}")
print(f"Bodies: {model.nbody}")
```

## URDF Details

The robot URDF (`g1_comp_dex1_1_backpack.urdf`) includes:
- Kinematic chain for the full robot body
- Joint definitions with limits and damping parameters
- 3D mesh files for visualization
- Collision geometry definitions

To inspect joint details:
```python
spec = mujoco.MjSpec.from_string(urdf_text)
for joint in spec.joints:
    print(f"Joint: {joint.name}, Type: {joint.type}, Range: {joint.range}")
```

## Calibration Data

Calibration files in the `calibration/` directory contain:
- Joint offsets and biases
- Sensor calibration parameters
- End-effector transformations

## Sensors

### RGB-D Cameras
| Sensor | Location | Serial Number | Computer | Notes |
|--------|----------|---------------|----------|-------|
| D435 | Head | 344422071022 | Orin | Main head camera |
| D455 | Head | 242422302279 | Thor | Alternative head camera |
| D405 | Left Wrist | 352122271203 | Thor | Wrist manipulation |
| D405 | Right Wrist | 335122271428 | Thor | Wrist manipulation |

### Lidar (Optional)
- **Model**: Livox MID360
- **Config**: `sensors/MID360_config.json`

## Branch Structure

Each branch in this repository represents a different robot instance:
- `main`: Base template
- `saturn`: Current robot instance (Unitree G1-EDU-U2)
- Other branches: Additional robot instances

To switch branches:
```bash
git switch [branch-name]
```

## Troubleshooting

### Viewer doesn't start
- Ensure MuJoCo is properly installed: `pip install -U mujoco`
- Check that the URDF path exists and is accessible
- Verify mesh files are in the correct location

### Mesh loading errors
- The script automatically adjusts `meshdir` in the URDF for proper path resolution
- Ensure mesh files are in `urdf/meshes/`

### Joint slider issues
- Only hinge (revolute) and slide (prismatic) joints are shown as sliders
- Fixed joints and free joints are not actuated

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [URDF Format Specification](http://wiki.ros.org/urdf)
- [Unitree G1 Robot](https://www.unitreerobotics.com/)

## License

This project is part of the Ether robotics platform.
