import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import threading
from queue import Queue, Empty
from typing import Optional, Tuple
from foodninja.core.models import SearchWindow, Measurement
from foodninja.initialization.interfaces import InitializationResult

class HandDetector:
    """
    Uses the modern MediaPipe Tasks API to detect hands for initial state or recovery.
    """
    def __init__(self, model_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        # Configure Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_width = 0
        self.last_height = 0
 
    def detect_hand(self, frame: np.ndarray, roi: Tuple[int, int, int, int] | None = None) -> Optional[Measurement]:
        """
        Runs MediaPipe Tasks inference to find the hand.
        Optional ROI (x, y, w, h) can be provided to focus search.
        """
        h, w, _ = frame.shape
        self.last_width, self.last_height = w, h
        
        target_frame = frame
        offset_x, offset_y = 0, 0
        
        if roi:
            rx, ry, rw, rh = roi
            # Ensure ROI is within bounds
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(rw, w - rx), min(rh, h - ry)
            target_frame = frame[ry:ry+rh, rx:rx+rw]
            offset_x, offset_y = rx, ry

        # Convert OpenCV BGR frame to RGB and then to MediaPipe Image
        rgb_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform detection
        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None

        # Get the first hand
        hand_landmarks = results.hand_landmarks[0]
        
        # Calculate bounding box from landmarks
        target_h, target_w, _ = target_frame.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        
        min_x, max_x = min(x_coords) * target_w, max(x_coords) * target_w
        min_y, max_y = min(y_coords) * target_h, max(y_coords) * target_h
        
        # Add some padding
        padding = 15
        bbox_w = (max_x - min_x) + padding * 2
        bbox_h = (max_y - min_y) + padding * 2
        cx = (min_x + max_x) / 2 + offset_x
        cy = (min_y + max_y) / 2 + offset_y

        # Access handedness for confidence (score of the first classification)
        confidence = 0.0
        if results.handedness:
            confidence = results.handedness[0][0].score

        return Measurement(
            cx=cx,
            cy=cy,
            width=float(bbox_w),
            height=float(bbox_h),
            confidence=confidence
        )

    def close(self):
        """Clean up the landmarker."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    def __del__(self):
        self.close()


class ThreadedHandDetector:
    """
    Wrapper for HandDetector that runs MediaPipe in a background thread.
    Prevents blocking the main game/tracking loop.
    """
    def __init__(self, model_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.model_path = model_path
        self.min_det = min_detection_confidence
        self.min_track = min_tracking_confidence
        
        self.input_queue = Queue(maxsize=1)
        self.result_lock = threading.Lock()
        self.latest_result: Optional[Measurement] = None
        self.new_result_available = False
        self.running = True
        
        # Start background thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background thread worker logic."""
        # MediaPipe landmarker MUST be created inside the thread it is used in
        detector = HandDetector(self.model_path, self.min_det, self.min_track)
        
        while self.running:
            try:
                # Wait for a frame to process
                item = self.input_queue.get(timeout=0.1)
                if item is None: break # Shutdown signal
                
                frame, roi = item
                result = detector.detect_hand(frame, roi)
                
                with self.result_lock:
                    self.latest_result = result
                    self.new_result_available = True
                    
                self.input_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"ThreadedHandDetector worker error: {e}")
                continue
        
        detector.close()

    def enqueue_frame(self, frame: np.ndarray, roi: Tuple[int, int, int, int] | None = None):
        """Non-blocking: Enqueue a frame if the detector is ready."""
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait() # Drop oldest frame if stale
            except Empty:
                pass
        self.input_queue.put((frame.copy(), roi))

    def get_latest_result(self) -> Tuple[bool, Optional[Measurement]]:
        """Non-blocking: Check for a new result."""
        with self.result_lock:
            available = self.new_result_available
            result = self.latest_result
            self.new_result_available = False
            return available, result

    def close(self):
        """Stop the background thread."""
        self.running = False
        self.input_queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def __del__(self):
        self.close()
