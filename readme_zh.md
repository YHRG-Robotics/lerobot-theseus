# 安装
我们已将S1 SDK集成到lerobot内可直接使用pip绑定安装
```shell
conda create -y -n lerobot python=3.11
pip install -e .
```
在[config](lerobot-theseus/lerobot/common/robot_devices/robots/configs.py)中，我们给了theseus_s1的四臂配置样例
```python
@RobotConfig.register_subclass("theseus_S1")
@dataclass
class Theseus_S1RobotConfig(TheseusManipulatorRobotConfig):
    leader_arms: dict[str, ArmConfig] = field(
        default_factory=lambda: {
            "part1": S1ArmConfig(
                mode="only_real",
                dev="PCAN_USBBUS1",
                end_effector="teach"
            ),
            "part2": S1ArmConfig(
                mode="only_real",
                dev="PCAN_USBBUS2",
                end_effector="teach"
            )
        }
    )

    follower_arms: dict[str, ArmConfig] = field(
        default_factory=lambda: {
            "part1": S1ArmConfig(
                mode="only_real",
                dev="PCAN_USBBUS3",
                end_effector="gripper"
            ),
            "part2": S1ArmConfig(
                mode="only_real",
                dev="PCAN_USBBUS4",
                end_effector="gripper"
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left": OpenCVCameraConfig(
                camera_index=1, fps=30, width=640, height=480, rotation=90
            ),
            "right": OpenCVCameraConfig(
                camera_index=1, fps=30, width=640, height=480, rotation=90
            ),
            "middle": OpenCVCameraConfig(
                camera_index=2, fps=30, width=640, height=480, rotation=90
            ),
            
        }
    )
```
注意，主从摇操时，每组臂的key需要保持一致。另外，在cameras中，fps必须小于等于摄像头的最大fps，否则可能会导致录制视频压缩。  
举例：如果摄像头的fps只有10，但此处的fps设置为30，录制时间为10s，则数据集中录制的视频时间仅为3.3s。  
我们已完全兼容了lerobot平台，具体使用方法可参考[example](examples)

