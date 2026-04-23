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
        self.is_initial = True
        self.current_state: Optional[TrackingState] = None
        self.visual_state: Optional[TrackingState] = None
        self.history = deque(maxlen=config.trajectory_history_length)
        
        self.filter_x = OneEuroFilter(min_cutoff=config.one_euro_min_cutoff, beta=config.one_euro_beta)
        self.filter_y = OneEuroFilter(min_cutoff=config.one_euro_min_cutoff, beta=config.one_euro_beta)

    def process_frame(self, frame: np.ndarray, dt: float = 1.0/60.0) -> TrackingStepResult:
        """
        Step through the tracking pipeline for a single frame.
        dt: actual elapsed time since last frame — keeps Kalman F accurate.
        """
        self.kf.update_dt(max(0.005, min(dt, 0.1)))

        # ── 1. Check for MediaPipe result from background thread ──────────────
        new_ready, ground_truth = self.detector.get_latest_result()

        # Submit next frame to background (every N frames or when lost)
        if self.mode == TrackMode.REINIT or \
                self.frames_since_reinit >= self.config.mediapipe_correction_interval_frames:
            roi = None
            if self.is_initial:
                fh, fw = frame.shape[:2]
                roi = (fw // 2 - 150, fh // 2 - 150, 300, 300)
            self.detector.enqueue_frame(frame, roi)
            self.frames_since_reinit = 0  # Reset counter so we don't flood the queue

        self.frames_since_reinit += 1

        # Apply MediaPipe ground-truth when it arrives
        if new_ready and ground_truth:
            if self.mode == TrackMode.TRACKING and self.current_state is not None:
                # ── Soft update: treat MediaPipe as a high-quality Kalman measurement.
                # Does NOT reset P — avoids the cold-start oscillation that a full
                # reinit causes every few frames.
                x_upd, _ = self.kf.update(ground_truth.to_array())
                self.current_state = TrackingState.from_array(x_upd)
                self.lost_count = 0
                # Refresh CamShift histogram with the new accurate position
                self.camshift.set_target(frame, (
                    int(ground_truth.cx - ground_truth.width / 2),
                    int(ground_truth.cy - ground_truth.height / 2),
                    int(ground_truth.width),
                    int(ground_truth.height)
                ))
            else:
                # Full reinit: Kalman state unknown (REINIT/LOST too long)
                return self._handle_reinit_result(frame, ground_truth)

        # ── 2. In REINIT mode with no ground-truth yet: return last known pos ─
        if self.mode == TrackMode.REINIT:
            return TrackingStepResult(
                mode=TrackMode.REINIT,
                state=self.visual_state if self.visual_state else TrackingState(0,0,0,0,0,0),
                lost_count=self.lost_count,
                accepted_measurement=False,
                frames_since_reinit=self.frames_since_reinit,
                confidence=0.0,
                trajectory=list(self.history)
            )

        # ── 3. Kalman Prediction ──────────────────────────────────────────────
        x_pred, P_pred = self.kf.predict()
        pred_state = TrackingState.from_array(x_pred)

        # ── 4. CamShift Measurement ───────────────────────────────────────────
        trace_p = np.trace(P_pred[:2, :2])
        z_t = self.camshift.measure(
            frame, pred_state, trace_p,
            velocity_lookahead=self.config.velocity_lookahead_factor,
            dt=self.kf._dt,
        )

        # ── 5. Gating + Kalman Update ─────────────────────────────────────────
        accepted = False
        if z_t is not None:
            accepted = accept_measurement(
                predicted_state=pred_state,
                measurement=z_t,
                position_variance_x=P_pred[0, 0],
                position_variance_y=P_pred[1, 1],
                config=self.config,
            )

        if accepted:
            x_upd, _ = self.kf.update(z_t.to_array())
            self.current_state = TrackingState.from_array(x_upd)
            self.lost_count = 0
            self.mode = TrackMode.TRACKING
            velocity_mag = np.sqrt(self.current_state.vx**2 + self.current_state.vy**2)
            if z_t.confidence > 0.7 and velocity_mag < 200:
                self.camshift.update_histogram(frame, self.current_state.to_roi(), alpha=0.05)
        else:
            # Use Kalman prediction as best estimate during loss
            self.current_state = pred_state
            self.lost_count += 1
            if self.lost_count >= self.config.lost_reinit_threshold:
                self.mode = TrackMode.REINIT
            else:
                self.mode = TrackMode.LOST

        # ── 6. Smooth for visualization (One Euro Filter) ─────────────────────
        if self.current_state:
            ts = time.time()
            smooth_x = self.filter_x.update(self.current_state.cx, ts)
            smooth_y = self.filter_y.update(self.current_state.cy, ts)
            self.visual_state = TrackingState(
                cx=smooth_x, cy=smooth_y,
                vx=self.current_state.vx, vy=self.current_state.vy,
                width=self.current_state.width, height=self.current_state.height,
            )
            self.history.append((self.visual_state.cx, self.visual_state.cy))

        return TrackingStepResult(
            mode=self.mode,
            state=self.visual_state if self.visual_state else TrackingState(0,0,0,0,0,0),
            lost_count=self.lost_count,
            accepted_measurement=accepted,
            frames_since_reinit=self.frames_since_reinit,
            confidence=z_t.confidence if z_t else 0.0,
            trajectory=list(self.history),
        )

    def _handle_reinit_result(self, frame: np.ndarray, measurement: Measurement) -> TrackingStepResult:
        """Full Kalman reinit from MediaPipe ground-truth (used only when REINIT/LOST)."""
        initial_x = np.array([
            measurement.cx, measurement.cy,
            0.0, 0.0,
            measurement.width, measurement.height,
        ], dtype=np.float32)
        self.kf.initialize(initial_x)
        self.camshift.set_target(frame, (
            int(measurement.cx - measurement.width / 2),
            int(measurement.cy - measurement.height / 2),
            int(measurement.width),
            int(measurement.height),
        ))
        self.mode = TrackMode.TRACKING
        self.lost_count = 0
        self.frames_since_reinit = 0
        self.is_initial = False
        self.current_state = TrackingState.from_array(initial_x)
        self.visual_state = self.current_state
        return TrackingStepResult(
            mode=TrackMode.REINIT,
            state=self.visual_state,
            lost_count=0,
            accepted_measurement=True,
            frames_since_reinit=0,
            confidence=measurement.confidence,
            trajectory=list(self.history),
        )


