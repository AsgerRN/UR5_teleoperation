import time
import numpy as np

from ur5 import UR5_Dummy, UR5_Pi0
from vr import VR
from datasaver import DataSaver, DataSaverConcurrent

def main():
    text_instruction = "Pick up the red object"
    print(f"Language instruction: {text_instruction}")

    ur_ip = "10.242.15.100"
    print(f"Connecting to UR5e at {ur_ip}...")
    robot = UR5_Pi0(ur_ip)

    vr = VR(robot.pose)
    data_saver = DataSaverConcurrent(save_folder="/mnt/big/teleop_data")

    
    print("\n========================================")
    print("             TELEOP ACTIVE              ")
    print("========================================")

    try:
        current_state = robot.pose
        
        recording = False
        last_b_pressed = False

        while True:
            loop_start_time = time.perf_counter()

            # --- TIMED SUB-COMPONENTS ---
            t0 = time.perf_counter()
            current_state = robot.pose
            t_pose = time.perf_counter() - t0

            t0 = time.perf_counter()
            target_pose = vr.get_action()
            t_action = time.perf_counter() - t0

            t0 = time.perf_counter()
            camera_frame = robot.camera
            t_cam = time.perf_counter() - t0

            # --- TOGGLE RECORDING LOGIC ---
            b_pressed = vr.is_handoff_pressed() 
            
            if b_pressed and not last_b_pressed:
                if not recording:
                        print("🚀 RECORDING STARTED")
                        recording = True
                else:
                    recording = False
                    data_saver.save_episode()
                    print("✅ Episode saved.")
                        
                    
            last_b_pressed = b_pressed

            t_add = 0.0
            if recording:
                t0 = time.perf_counter()
                data_saver.add_frame(
                    robot_state=current_state,
                    action=target_pose,
                    camera_frame=camera_frame,
                    language_instruction=text_instruction
                )
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

if __name__ == "__main__":
    main()