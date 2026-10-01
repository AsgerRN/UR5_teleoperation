import time
import os
from typing import Any
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
import numpy as np
from numpy.typing import NDArray

import queue
import threading

import av

av.logging.set_level(av.logging.ERROR)

class DataSaver:
    def __init__(self, save_folder: str|Path, repo_id: str = "local/ur5e_pi0_dataset"):
        self.recording = False
        
        features: dict[str, Any] = {
                "observation.state": {"dtype": "float32", "shape": (7,)},
                "action": {"dtype": "float32", "shape": (7,)},
                "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
                "language_instruction": {"dtype": "string", "shape": (1,)},    
            }

        os.makedirs(save_folder, exist_ok=True)
        save_dir = os.path.join(save_folder, f"ur5e_pi0_dataset_{time.strftime('%Y%m%d_%H%M%S')}")

        self.dataset: LeRobotDataset = LeRobotDataset.create( # type: ignore
            repo_id=repo_id, 
            fps=50, 
            features=features,
            root=save_dir,
            rgb_encoder=RGBEncoderConfig(vcodec="h264")
        )

    def __del__(self):
        if self.recording:
            self.dataset.save_episode() # type: ignore

    def add_frame(self, robot_state: NDArray[np.float32], action: NDArray[np.float32], camera_frame: NDArray[np.uint8], language_instruction: str) -> None:
        
        self.dataset.add_frame({ # type: ignore
            "observation.state": robot_state,
            "action": action,
            "observation.images.wrist": camera_frame,
            "language_instruction": language_instruction,
            "task": language_instruction,
        })

        self.recording = True

    def save_episode(self) -> None:
        if self.recording:
            self.dataset.save_episode() # type: ignore
            self.recording = False

class DataSaverConcurrent(DataSaver):
    def __init__(self, save_folder: str | Path, repo_id: str = "local/ur5e_pi0_dataset", queue_length: int = 50 * 60 * 5):
        super().__init__(save_folder, repo_id)

        self.queue_length = queue_length
        self._queue: queue.Queue[tuple[str, dict[str, Any]|None]] = queue.Queue(maxsize=queue_length) # 50 fps * 5 minutes
        self._emptying_full_queue = threading.Event()
        
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        recording = False
        while True:
            command, data = self._queue.get()
            try:
                match command:
                    case "frame":
                        self.dataset.add_frame(data) # type: ignore
                        recording = True
                    
                    case "save":
                        if not recording:
                            print("⚠️ No frames recorded. Nothing to save.")
                            continue

                        print("💾 Saving episode...")
                        self.dataset.save_episode() # type: ignore
                        recording = False
                        print("💾 Episode saved successfully.")
                
                    case "shutdown":
                        if recording:
                            print("💾 Saving episode before shutdown...")
                            self.dataset.save_episode() # type: ignore
                            print("💾 Episode saved successfully.")
                        self._queue.task_done()
                        break # Exit the infinite loop cleanly

                    case _:
                        raise ValueError(f"Unknown command: {command}")

            except Exception as e:
                print(f"DataSaver worker error: {e}")
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    print("✅ Queue has been emptied. ")
        
        print("DataSaver worker thread has exited.")

    def add_frame(self, robot_state: NDArray[np.float32], action: NDArray[np.float32], camera_frame: NDArray[np.uint8], language_instruction: str) -> None:
        if self._queue.qsize() >= self._queue.maxsize * 0.8:
            print(f"⚠️ DataSaver queue is almost full {self._queue.qsize() / self._queue.maxsize * 100:.1f}%")
        if self._emptying_full_queue.is_set():
            if self._queue.empty():
                self._emptying_full_queue.clear()
                print("✅ Queue has been emptied. Resuming frame recording.")
            else:
                print(f"⚠️ Processing full queue. Waiting for the queue to be emptied before adding new frames {self._queue.qsize() / self._queue.maxsize * 100:.1f}%.")
            return
        
        data_dict: dict[str, Any] = {
            "observation.state": robot_state,
            "action": action,
            "observation.images.wrist": camera_frame,
            "language_instruction": language_instruction,
            "task": language_instruction,
        }

        try:
            self._queue.put_nowait(("frame", data_dict))
        except queue.Full:
            print("⛔️ DataSaver queue is full. Stopping recording till all frames are saved.")
            self._queue.put(("save", None))
            self._emptying_full_queue.set()

    def save_episode(self) -> None:
        if self._emptying_full_queue.is_set():
            print("⚠️ Queue is currently being emptied. Episode is automatically saved.")
            return

        print(f"💾 Saving episode... {(self._queue.maxsize - self._queue.qsize()) / 50:.1f} seconds left in the queue")
        self._queue.put(("save", None))

    def __del__(self) -> None:
        if self._worker_thread.is_alive():
            print("Shutting down DataSaver...")
            self._queue.put(("shutdown", None))
            self._worker_thread.join(timeout=self.queue_length)
        super().__del__()

    # --- Properties ---
    
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_empty(self) -> bool:
        return self._queue.empty()

