import cv2
import numpy as np
from typing import Tuple, Optional
from foodninja.core.models import CovarianceState, Measurement, SearchWindow, TrackingState

class CamShiftModule:
    """
    Advanced CamShift module with H-S 2D Histograms, Preprocessing, 
    Multi-metric Confidence, and Spatial Priors.
    """
    def __init__(self, base_window_add: float = 20.0, k_uncertainty: float = 2.0):
        self.roi_hist = None
        self.base_window_add = base_window_add
        self.k_uncertainty = k_uncertainty
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # Morphology kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def set_target(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """Sets up the H-S histogram for the target region."""
        x, y, w, h = roi
        roi_img = frame[int(y):int(y+h), int(x):int(x+w)]
        if roi_img.size == 0:
            return

        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Mask for skin-like colors (Hue: 0-180, Saturation: 60-255, Value: 32-255)
        mask = cv2.inRange(hsv_roi, np.array((0., 50., 40.)), np.array((180., 255., 255.)))
        
        # Calculate 2D Hist: Hue (Channel 0) and Saturation (Channel 1)
        self.roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    def measure(self, frame: np.ndarray, predicted_state: TrackingState, uncertainty_trace: float, velocity_lookahead: float = 0.0, dt: float = 1.0/60.0) -> Optional[Measurement]:
        """
        Executes CamShift with advanced confidence scoring and spatial prior.
        velocity_lookahead: fraction of (vx*dt, vy*dt) to shift the search window
                            center ahead of the Kalman prediction.
        """
        if self.roi_hist is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Backprojection with H-S
        dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        
        # 2. Preprocessing backprojection
        dst = cv2.GaussianBlur(dst, (5, 5), 0)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel_open)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel_close)
        _, dst = cv2.threshold(dst, 10, 255, cv2.THRESH_TOZERO) # Loosened from 20

        # 3. Calculate dynamic search window, optionally shifted along velocity
        search_cx = predicted_state.cx + velocity_lookahead * predicted_state.vx * dt
        search_cy = predicted_state.cy + velocity_lookahead * predicted_state.vy * dt
        growth = self.base_window_add + self.k_uncertainty * (uncertainty_trace ** 0.5)
        sw_w = max(predicted_state.width, 40) + growth
        sw_h = max(predicted_state.height, 40) + growth
        
        sw_x = int(search_cx - sw_w / 2)
        sw_y = int(search_cy - sw_h / 2)
        
        fh, fw = frame.shape[:2]
        track_window = (
            max(0, min(fw - 1, sw_x)), 
            max(0, min(fh - 1, sw_y)), 
            max(1, min(fw - sw_x, int(sw_w))), 
            max(1, min(fh - sw_y, int(sw_h)))
        )

        try:
            ret, _ = cv2.CamShift(dst, track_window, self.term_crit)
            (cx, cy), (w, h), angle = ret
            
            # 4. Multi-Metric Confidence Calculation
            pts = cv2.boxPoints(ret)
            pts = np.int32(pts)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            
            # Clamp bounding rect to frame
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(rw, fw - rx), min(rh, fh - ry)
            
            if rw <= 0 or rh <= 0:
                return None
                
            roi_bp = dst[ry:ry+rh, rx:rx+rw]
            
            # Metric 1: Mean Intensity (Density)
            mean_score = np.mean(roi_bp) / 255.0
            
            # Metric 2: Peak Intensity (Signal strength)
            _, max_val, _, _ = cv2.minMaxLoc(roi_bp)
            peak_score = max_val / 255.0
            
            # Metric 3: Active Pixel Ratio (Structure)
            active_pixels = np.count_nonzero(roi_bp > 40)
            area_ratio = active_pixels / (rw * rh)
            
            # Blended Confusion Score (Weights from log: 0.3, 0.4, 0.3)
            confidence = (0.3 * mean_score) + (0.4 * peak_score) + (0.3 * area_ratio)
            
            # 5. Spatial Prior Penalty (Center Penalty)
            # Distance from velocity-shifted prediction center
            dist_sq = ((cx - search_cx)**2 + (cy - search_cy)**2)
            # Soften penalty: use larger sigma
            sigma_sq = (sw_w * 0.7)**2 
            spatial_penalty = np.exp(-dist_sq / (2 * sigma_sq))
            
            confidence *= spatial_penalty
            
            return Measurement(
                cx=float(cx),
                cy=float(cy),
                width=float(w),
                height=float(h),
                confidence=float(confidence)
            )
        except Exception:
            return None

    def update_histogram(self, frame: np.ndarray, roi: Tuple[int, int, int, int], alpha: float = 0.05):
        """Slowly blends a new histogram into the current model (EMA)."""
        if self.roi_hist is None:
            return
            
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        # Clamp ROI
        rx, ry = max(0, int(x)), max(0, int(y))
        rw, rh = min(int(w), fw - rx), min(int(h), fh - ry)
        
        roi_img = frame[ry:ry+rh, rx:rx+rw]
        if roi_img.size == 0:
            return

        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 50., 40.)), np.array((180., 255., 255.)))
        new_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)
        
        # EMA update
        self.roi_hist = (1 - alpha) * self.roi_hist + alpha * new_hist
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)


def build_search_window(
    state: TrackingState,
    covariance: CovarianceState,
    config,
) -> SearchWindow:
    """Build a CamShift search window sized by Kalman uncertainty.

    Used by the standalone tracking demo (``tracking_demo.py``).
    """
    uncertainty = (covariance.pos_x_var + covariance.pos_y_var) ** 0.5
    growth = config.base_search_window + config.search_window_scale * uncertainty
    w = max(state.width, 40) + growth
    h = max(state.height, 40) + growth
    return SearchWindow(
        x=state.cx - w / 2,
        y=state.cy - h / 2,
        width=w,
        height=h,
    )
