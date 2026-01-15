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


def make_s1_arm_from_configs(arm_config: S1ArmConfig) -> S1_arm:
    control_dict = {
        "only_real": control_mode.only_real,
        "only_sim": control_mode.only_sim,
    }
    
    # print(f"Creating S1 arm with config: {[arm_config['main'].keys()]}")
    return S1_arm(
        mode=control_dict[arm_config['main'].mode],
        dev=arm_config['main'].dev,
        end_effector=arm_config['main'].end_effector,
        arm_version="V1"
    )