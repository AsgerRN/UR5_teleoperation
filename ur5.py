import cv2
import numpy as np
from typing import Any
from numpy.typing import NDArray
from rtde_control import RTDEControlInterface  # type: ignore
from rtde_receive import RTDEReceiveInterface  # type: ignore
from gripper import UR5_RG2
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


class UR5:
    def __init__(self, ur_ip: str):
        self.c: Any = RTDEControlInterface(ur_ip)
        self.r: Any = RTDEReceiveInterface(ur_ip)

        self.gripper = UR5_RG2(ur_ip)
        self.camera_pipeline = self._init_camera()

    def __del__(self):
        if getattr(self, "c", None) is not None:
            self.c.disconnect()
        if getattr(self, "r", None) is not None:
            self.r.disconnect()
        if getattr(self, "camera_pipeline", None) is not None:
            self.camera_pipeline.stop()

    def _ur_pose(self) -> NDArray[np.float64]:
        """Raw physical UR TCP pose."""
        return np.asarray(self.r.getActualTCPPose(), dtype=np.float64)

    @property
    def pose(self) -> NDArray[np.float32]:
        """Pose in UR base coordinates."""
        return self._ur_pose().astype(np.float32)

    def move(self, pose: NDArray[np.float64], speed: float = 0.02, acceleration: float = 0.02) -> None:
        """Move to an absolute pose in UR base coordinates."""
        if pose.shape != (6,):
            raise ValueError(f"Expected UR pose shape (6,), got {pose.shape}")

        self.c.moveL(pose.tolist(), speed, acceleration)

    def _init_camera(self) -> Pipeline:
        pipeline = Pipeline()
        config = Config()

        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # Requesting 640x480 at 60 FPS (which comfortably feeds our 50Hz loop)
            color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 60)
            config.enable_stream(color_profile)
            pipeline.start(config)
            print("[INFO] Orbbec Gemini 305 initialized via SDK at 640x480 @ 60FPS")
        except Exception as e:
            print(f"[WARNING] Could not set 60Hz profile: {e}. Falling back to default.")
            pipeline.start()

        return pipeline
    
    def _camera_process_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return frame

    @property
    def camera(self):
        # Wait for frame with a tight 15ms timeout
        for _ in range(3):
            frames = self.camera_pipeline.wait_for_frames(100)
            if frames:
                break
            print("⚠️ Waiting for camera frame...")
        else:
            raise RuntimeError("Failed to retrieve frames from Orbbec camera")
            
        color_frame = frames.get_color_frame()            
        data = color_frame.get_data()
        frame_rgb = np.frombuffer(data, dtype=np.uint8).reshape((color_frame.get_height(), color_frame.get_width(), 3))
        
        return self._camera_process_frame(frame_rgb)


class UR5_Pi0(UR5):

    def __init__(self, ur_ip: str):
        super().__init__(ur_ip)

    @property
    def pose(self) -> NDArray[np.float32]:
        """
        Current state in Pi0 format:
        [j0, j1, j2, j3, j4, j5, gripper_open]
        """
        joints = self.r.getActualQ()
        gripper = float(np.clip(self.gripper.get_width() / 110.0, 0.0, 1.0))
        return np.array([*joints, gripper], dtype=np.float32)

    def _camera_process_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # Convert RGB to BGR for OpenCV compatibility
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_contiguous = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        return frame_contiguous

    def move(self, pose: NDArray[np.float64|np.float32], speed: float = 0.5, acceleration: float = 0.5) -> None:
        """
        Execute a Pi0 action:
        [j0_target, j1_target, j2_target, j3_target, j4_target, j5_target, gripper_target]
        """
        target_joints = pose[:6]
        gripper_target = pose[6]

        target_width = np.clip(gripper_target * 110, 0, 100)
        self.gripper.overwrite_move(target_width)
        
        # Stream smoothly using servoJ with positional arguments:
        # self.c.servoJ(joints, velocity, acceleration, dt, lookahead_time, gain)
        self.c.servoJ(
            target_joints.tolist(), 
            0.0,    # velocity (not used in servoJ, usually 0)
            0.0,    # acceleration (not used in servoJ, usually 0)
            0.02,   # dt (matches your 50Hz control loop)
            0.1,    # lookahead_time (trajectory smoothing window)
            300     # gain (proportional controller gain)
        )


class UR5_Dummy(UR5):
    def __init__(self, _: str):
        self.c = None
        self.r = None
        self.gripper = None
        self.camera_pipeline = self._init_camera()

    @property
    def pose(self) -> NDArray[np.float32]:
        return np.zeros((8,), dtype=np.float32)

    def move(self, pose: NDArray[np.float64|np.float32], speed: float = 0.02, acceleration: float = 0.02) -> None:
        print("Dummy pose:", pose)


if __name__ == "__main__":
    ur = UR5_Pi0("10.242.15.100")

    print("pose:")
    print(ur.pose)

    pi0_action = np.array([
        0.00,  # j0
        0.00,  # j1
        0.00,  # j2
        0.00,  # j3
        0.00,  # j4
        0.00,  # j5
        1.00,  # gripper
    ])

    ur.move(pi0_action)
