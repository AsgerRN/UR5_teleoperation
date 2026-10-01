import asyncio
import websockets
import json

async def print_simplified_vr():
    uri = "ws://127.0.0.1:8443/ws"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Squeeze triggers/grips to test.\n")
            
            while True:
                # This now waits for the exact moment the next frame arrives
                message = await websocket.recv()
                data = json.loads(message)
                
                controllers = data.get("controllers", {})
                
                # --- Right Controller ---
                right = controllers.get("right", {})
                r_pos = right.get("position", [0.0, 0.0, 0.0])
                r_buttons = right.get("buttons", [])
                
                r_trigger = r_buttons[0]["v"] if len(r_buttons) > 0 else 0.0
                r_grip = r_buttons[1]["v"] if len(r_buttons) > 1 else 0.0
                
                # --- Left Controller ---
                left = controllers.get("left", {})
                l_pos = left.get("position", [0.0, 0.0, 0.0])
                l_buttons = left.get("buttons", [])
                
                l_trigger = l_buttons[0]["v"] if len(l_buttons) > 0 else 0.0
                l_grip = l_buttons[1]["v"] if len(l_buttons) > 1 else 0.0

                # Clear screen and print formatted status
                print("\033[H\033[J", end="") 
                print("==================================================")
                print("            VR CONTROLLER LIVE STATUS             ")
                print("==================================================")
                print("RIGHT CONTROLLER (Main Teleop):")
                print(f"  Position [X, Y, Z]: [{r_pos[0]:.3f}, {r_pos[1]:.3f}, {r_pos[2]:.3f}]")
                print(f"  Trigger (Gripper):  {r_trigger:.2f}  {'[PULLED]' if r_trigger > 0.1 else '[OPEN]'}")
                print(f"  Grip    (Clutch):   {r_grip:.2f}  {'[ENGAGED]' if r_grip > 0.1 else '[OFF]'}")
                print("--------------------------------------------------")
                print("LEFT CONTROLLER:")
                print(f"  Position [X, Y, Z]: [{l_pos[0]:.3f}, {l_pos[1]:.3f}, {l_pos[2]:.3f}]")
                print(f"  Trigger:            {l_trigger:.2f}")
                print(f"  Grip:               {l_grip:.2f}")
                print("==================================================")
                print("Press CTRL+C to exit")
                
                # REMOVED: await asyncio.sleep(0.05)

    except websockets.exceptions.ConnectionClosed:
        print("\nConnection to relay closed.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(print_simplified_vr())