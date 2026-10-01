import pathlib
import os
from vr_teleop_kit.lerobot import SingleArmQuestTeleoperator, SingleArmQuestTeleoperatorConfig
import numpy as np
from numpy.typing import NDArray

class VR:
    def __init__(self, robot_pose: NDArray[np.float32]):
        if "DK1_URDF" not in os.environ:
            project_root = pathlib.Path(__file__).parent.resolve()
            os.environ["DK1_URDF"] = str(project_root / "models" / "ur5e" / "ur5e.xml")

        config = SingleArmQuestTeleoperatorConfig(
                arm="right",
                ws_url="ws://127.0.0.1:8443/ws",
                publish_ik_state=False,
                rest_qpos_right=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
                scale_translation=0.01,   
                scale_rotation=0.001,      
                max_dq_per_joint_scalar_pos=0.0015,  
                max_dq_per_joint_scalar_rot=0.005,   
        )
        self.teleop = SingleArmQuestTeleoperator(config)
        self.teleop.connect()

        padded_qpos = np.zeros(8, dtype=np.float64)
        padded_qpos[:6] = robot_pose[:6]
        self.teleop._inner._arms["right"]["qpos"] = padded_qpos

    def __del__(self):
        if hasattr(self, 'teleop'):
            self.teleop.disconnect()

    def get_action(self, robot_pose: NDArray[np.float32]|None = None) -> NDArray[np.float32]:
        action_dict = self.teleop.get_action()

        if robot_pose is None:
            robot_pose = np.full((7,), np.nan, dtype=np.float32)

        target_pose = np.array([
            action_dict.get("joint_1.pos", robot_pose[0]),
            action_dict.get("joint_2.pos", robot_pose[1]),
            action_dict.get("joint_3.pos", robot_pose[2]),
            action_dict.get("joint_4.pos", robot_pose[3]),
            action_dict.get("joint_5.pos", robot_pose[4]),
            action_dict.get("joint_6.pos", robot_pose[5]),
            action_dict.get("gripper.pos", robot_pose[6])
        ], dtype=np.float32)
        
        if np.any(np.isnan(target_pose)):
            raise ValueError("Received NaN values in action_dict, cannot compute target_pose.")

        return target_pose

    def is_handoff_pressed(self) -> bool:
        return self.teleop.is_handoff_pressed()

