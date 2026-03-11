import abc
from dataclasses import dataclass
from S1_SDK import control_mode, S1_arm
import draccus


@dataclass
class ArmConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@ArmConfig.register_subclass("s1")
@dataclass
class S1ArmConfig(ArmConfig):
    mode: str
    dev: str
    end_effector: str
    version: str


def make_s1_arm_from_configs(arm_configs: dict[str, S1ArmConfig], arm_name: str) -> S1_arm:
    control_dict = {
        "only_real": control_mode.only_real,
        "only_sim": control_mode.only_sim,
    }
    
    # print(f"Creating S1 arm with config: {[arm_configs[arm_name].keys()]}")
    return S1_arm(
        mode=control_dict[arm_configs[arm_name].mode],
        dev=arm_configs[arm_name].dev,
        end_effector=arm_configs[arm_name].end_effector,
        arm_version=arm_configs[arm_name].version,
    )