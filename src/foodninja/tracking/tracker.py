import numpy as np
from typing import Optional, Tuple
from collections import deque
import time
from foodninja.core.models import TrackMode, TrackingState, Measurement, TrackingStepResult
from foodninja.core.utils import get_resource_path
from foodninja.config import TrackingConfig
from .kalman_filter import KalmanFilterModule
from .camshift import CamShiftModule
from .gating import accept_measurement
from .filters import OneEuroFilter
from foodninja.initialization.mediapipe_recovery import HandDetector, ThreadedHandDetector

class HandTracker:
    """
    Orchestrates the tracking pipeline:
    Kalman Predict -> CamShift Measure -> Gating -> Kalman Update -> State Machine Management.
    Includes periodic Re-initialization via MediaPipe.
    """
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.mode = TrackMode.REINIT
        self.kf = KalmanFilterModule()
        self.camshift = CamShiftModule()
        self.detector = ThreadedHandDetector(model_path=get_resource_path(config.mediapipe_model_path))
        
        self.lost_count = 0
        self.frames_since_reinit = 0
        self.is_initial = True # Initially searching in calibration zone
        self.current_state: Optional[TrackingState] = None
        self.visual_state: Optional[TrackingState] = None
        self.history = deque(maxlen=config.trajectory_history_length)
        
        self.filter_x = OneEuroFilter(min_cutoff=config.one_euro_min_cutoff, beta=config.one_euro_beta)
        self.filter_y = OneEuroFilter(min_cutoff=config.one_euro_min_cutoff, beta=config.one_euro_beta)

    def process_frame(self, frame: np.ndarray) -> TrackingStepResult:
        """
        Step through the tracking pipeline for a single frame.
        """
        # 1. Background MediaPipe Handlers
        # Check for new ground-truth result from background thread
        new_ready, ground_truth = self.detector.get_latest_result()
        
        # Periodic submission to background
        if self.mode == TrackMode.REINIT or self.frames_since_reinit >= self.config.mediapipe_correction_interval_frames:
            roi = None
            if self.is_initial:
                fh, fw = frame.shape[:2]
                roi = (fw // 2 - 150, fh // 2 - 150, 300, 300)
            self.detector.enqueue_frame(frame, roi)

        # Apply ground truth if it just arrived
        if new_ready and ground_truth:
            return self._handle_reinit_result(frame, ground_truth)

        # In REINIT mode, continue UI without blocking
        if self.mode == TrackMode.REINIT:
            self.frames_since_reinit += 1
            return TrackingStepResult(
                mode=TrackMode.REINIT,
                state=self.visual_state if self.visual_state else TrackingState(0,0,0,0,0,0),
                lost_count=self.lost_count,
                accepted_measurement=False,
                frames_since_reinit=self.frames_since_reinit,
                confidence=0.0,
                trajectory=list(self.history)
            )

        # 2. Kalman Prediction
        x_pred, P_pred = self.kf.predict()
        pred_state = TrackingState.from_array(x_pred)
        
        # 3. CamShift Measurement
        trace_p = np.trace(P_pred[:2, :2])
        z_t = self.camshift.measure(frame, pred_state, trace_p)
        
        accepted = False
        if z_t is not None:
            # 4. Gating (Mahalanobis Distance)
            # Using pos_x_var and pos_y_var from P_pred diagonal
            accepted = accept_measurement(
                predicted_state=pred_state,
                measurement=z_t,
                position_variance_x=P_pred[0, 0],
                position_variance_y=P_pred[1, 1],
                config=self.config
            )

        if accepted:
            # 4. Standard Kalman Update (Official Step 4a)
            x_upd, P_upd = self.kf.update(z_t.to_array())
            self.current_state = TrackingState.from_array(x_upd)
            self.lost_count = 0
            self.mode = TrackMode.TRACKING
            
            # Dynamic Histogram Update can remain as an optimization
            velocity_mag = np.sqrt(self.current_state.vx**2 + self.current_state.vy**2)
            if z_t.confidence > 0.7 and velocity_mag < 200:
                self.camshift.update_histogram(frame, self.current_state.to_roi(), alpha=0.05)
        else:
            # 4. Calibration/Recovery Fallback
            # Use Prediction (Official requirement), but we've already 
            # submitted a frame to background above. 
            self.current_state = pred_state
            self.lost_count += 1
            self.mode = TrackMode.LOST if self.lost_count < self.config.lost_reinit_threshold else TrackMode.REINIT

        self.frames_since_reinit += 1

        # 6. Smooth for visualization (One Euro Filter)
        if self.current_state:
            ts = time.time()
            smooth_x = self.filter_x.update(self.current_state.cx, ts)
            smooth_y = self.filter_y.update(self.current_state.cy, ts)
            
            self.visual_state = TrackingState(
                cx = smooth_x,
                cy = smooth_y,
                vx = self.current_state.vx,
                vy = self.current_state.vy,
                width = self.current_state.width,
                height = self.current_state.height
            )
            self.history.append((self.visual_state.cx, self.visual_state.cy))

        return TrackingStepResult(
            mode=self.mode,
            state=self.visual_state if self.visual_state else TrackingState(0,0,0,0,0,0),
            lost_count=self.lost_count,
            accepted_measurement=accepted,
            frames_since_reinit=self.frames_since_reinit,
            confidence=z_t.confidence if z_t else 0.0,
            trajectory=list(self.history)
        )
        return TrackingStepResult(
            mode=self.mode,
            state=self.current_state,
            lost_count=self.lost_count,
            accepted_measurement=accepted,
            frames_since_reinit=self.frames_since_reinit
        )

    def _handle_reinit_result(self, frame: np.ndarray, measurement: Measurement) -> TrackingStepResult:
        """
        Applies a background MediaPipe result as a ground-truth anchor.
        """
        # Initialize Kalman filter with fresh ground truth
        initial_x = np.array([
            measurement.cx, measurement.cy, 
            0.0, 0.0, 
            measurement.width, measurement.height
        ], dtype=np.float32)
        self.kf.initialize(initial_x)
        
        # Setup CamShift histogram
        self.camshift.set_target(frame, (
            int(measurement.cx - measurement.width / 2),
            int(measurement.cy - measurement.height / 2),
            int(measurement.width),
            int(measurement.height)
        ))
        
        self.mode = TrackMode.TRACKING
        self.lost_count = 0
        self.frames_since_reinit = 0
        self.is_initial = False
        self.current_state = TrackingState.from_array(initial_x)
        self.visual_state = self.current_state
        
        return TrackingStepResult(
            mode=TrackMode.REINIT, # Keep UI informed it was a reinit frame
            state=measurement,
            lost_count=0,
            accepted_measurement=True,
            frames_since_reinit=0,
            confidence=measurement.confidence,
            trajectory=list(self.history)
        )
