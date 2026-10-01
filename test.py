import time
import os
import threading
import numpy as np
from pathlib import Path

from ur5 import UR5_Dummy, UR5_Pi0
from vr_teleop_kit.lerobot import SingleArmQuestTeleoperator, SingleArmQuestTeleoperatorConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

# Automatically set the URDF path relative to the script location
if "DK1_URDF" not in os.environ:
    project_root = Path(__file__).parent.resolve()
    os.environ["DK1_URDF"] = str(project_root / "models" / "ur5e" / "ur5e.xml")

def main():
    text_instruction = "Pick up the red object"
    print(f"Language instruction: {text_instruction}")

    ur_ip = "10.242.15.100"
    print(f"Connecting to UR5e at {ur_ip}...")
    robot = UR5_Pi0(ur_ip)
    
    print("Connecting to VR Relay Server...")
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
    teleop = SingleArmQuestTeleoperator(config)
    teleop.connect()

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,)},
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
        "language_instruction": {"dtype": "string", "shape": (1,)},    
    }
    dataset = LeRobotDataset.create(
        repo_id="local/ur5e_pi0_dataset", 
        fps=50, 
        features=features,
        root=f"/mnt/big/teleop_data/ur5e_pi0_dataset_{time.strftime('%Y%m%d_%H%M%S')}",
        rgb_encoder=RGBEncoderConfig(vcodec="h264")
    )
    
    print("\n========================================")
    print("             TELEOP ACTIVE              ")
    print("========================================")

    try:
        current_state = robot.pose
        
        # --- PADDED BYPASS ---
        padded_qpos = np.zeros(8, dtype=np.float64)
        padded_qpos[:6] = current_state[:6]
        teleop._inner._arms["right"]["qpos"] = padded_qpos

        recording = False
        is_saving = False      
        last_b_pressed = False

        while True:
            loop_start_time = time.perf_counter()

            # --- TIMED SUB-COMPONENTS ---
            t0 = time.perf_counter()
            current_state = robot.pose
            t_pose = time.perf_counter() - t0

            t0 = time.perf_counter()
            action_dict = teleop.get_action()
            t_action = time.perf_counter() - t0

            t0 = time.perf_counter()
            camera_frame = robot.camera
            t_cam = time.perf_counter() - t0

            # --- TOGGLE RECORDING LOGIC ---
            b_pressed = teleop.is_handoff_pressed() 
            
            if b_pressed and not last_b_pressed:
                if not recording:
                    if is_saving:
                        print("⏳ Still encoding previous video... please wait a second!")
                    else:
                        print("🚀 RECORDING STARTED")
                        recording = True
                else:
                    print("💥 RECORDING STOPPED - Saving episode in background...")
                    recording = False
                    is_saving = True  
                    
                    def save_worker():
                        nonlocal is_saving
                        dataset.save_episode()
                        print("\n✅ Episode saved! Safe to record the next one.")
                        is_saving = False  
                        
                    threading.Thread(target=save_worker).start()
                    
            last_b_pressed = b_pressed

            # Parse VR targets into our 7-DoF schema
            target_pose = np.array([
                action_dict.get("joint_1.pos", current_state[0]),
                action_dict.get("joint_2.pos", current_state[1]),
                action_dict.get("joint_3.pos", current_state[2]),
                action_dict.get("joint_4.pos", current_state[3]),
                action_dict.get("joint_5.pos", current_state[4]),
                action_dict.get("joint_6.pos", current_state[5]),
                action_dict.get("gripper.pos", current_state[6])
            ], dtype=np.float32)

            t_add = 0.0
            if recording:
                t0 = time.perf_counter()
                dataset.add_frame({
                    "observation.state": np.array(current_state, dtype=np.float32),
                    "action": target_pose,
                    "observation.images.wrist": camera_frame,
                    "language_instruction": text_instruction,
                    "task": text_instruction,
                })
                t_add = time.perf_counter() - t0

            t0 = time.perf_counter()
            robot.move(target_pose, speed=0.5, acceleration=0.5)
            t_move = time.perf_counter() - t0

            # --- PRECISION TIMING & PROFILING ---
            elapsed_time = time.perf_counter() - loop_start_time
            sleep_time = 0.02 - elapsed_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Detailed breakdown of where the extra time went
                print(f"⚠️ Loop dropped! Total: {elapsed_time:.3f}s | "
                      f"pose: {t_pose*1000:.1f}ms, action: {t_action*1000:.1f}ms, "
                      f"cam: {t_cam*1000:.1f}ms, add: {t_add*1000:.1f}ms, move: {t_move*1000:.1f}ms")

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        teleop.disconnect()
        if recording:
            print("Saving partial episode before exit...")
            dataset.save_episode()
        print("Done. Safe to exit.")

if __name__ == "__main__":
    main()