import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from foodninja.config import TrackingConfig
from foodninja.core.models import CovarianceState, Measurement, TrackMode, TrackingState
from foodninja.tracking.camshift import build_search_window
from foodninja.tracking.gating import squared_mahalanobis_distance
from foodninja.tracking.kalman import KalmanMotionModel
from foodninja.tracking.pipeline import advance_tracking_step
from foodninja.tracking.runtime import (
    blended_tracking_state,
    clamp_search_window,
    grow_covariance_after_rejection,
    measurement_from_window,
    shrink_covariance_after_acceptance,
)


@dataclass(slots=True)
class DemoSession:
    mode: TrackMode | None = None
    tracking_state: TrackingState | None = None
    covariance: CovarianceState | None = None
    histogram: Any | None = None
    track_window: tuple[int, int, int, int] | None = None
    lost_count: int = 0
    accepted_measurement: bool = False
    last_distance: float | None = None
    last_confidence: float = 0.0
    last_blend_weight: float = 0.0
    stable_accept_count: int = 0
    last_reinit_candidate: tuple[int, int, int, int] | None = None
    last_mediapipe_candidate: tuple[int, int, int, int] | None = None
    frame_index: int = 0
    unstable_frame_count: int = 0
    mediapipe_last_status: str = "idle"
    initialization_source: str = "manual"


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run `pip install -e .[vision]` before starting the demo."
        ) from exc
    return cv2


def _import_mediapipe() -> Any | None:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        return None
    return mp


def create_histogram_model(cv2: Any, frame: Any, roi: tuple[int, int, int, int]) -> Any:
    x, y, width, height = roi
    roi_bgr = frame[y : y + height, x : x + width]
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 35, 35), (180, 255, 255))
    histogram = cv2.calcHist([hsv_roi], [0, 1], mask, [32, 32], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    return histogram


def build_spatial_prior(
    frame_width: int,
    frame_height: int,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> Any:
    x_axis = np.arange(frame_width, dtype=np.float32)
    y_axis = np.arange(frame_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    sigma_x = max(8.0, sigma_x)
    sigma_y = max(8.0, sigma_y)
    exponent = (
        ((grid_x - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
        + ((grid_y - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
    )
    prior = np.exp(-exponent)
    return (255.0 * prior).astype(np.uint8)


def initialize_session(
    cv2: Any,
    frame: Any,
    selection: tuple[int, int, int, int],
    source: str = "manual",
) -> DemoSession:
    histogram = create_histogram_model(cv2, frame, selection)
    x, y, width, height = selection
    cx = x + width / 2.0
    cy = y + height / 2.0
    return DemoSession(
        mode=TrackMode.TRACKING,
        tracking_state=TrackingState(cx=cx, cy=cy, vx=0.0, vy=0.0, width=width, height=height),
        covariance=CovarianceState(
            pos_x_var=36.0,
            pos_y_var=36.0,
            vel_x_var=144.0,
            vel_y_var=144.0,
            width_var=49.0,
            height_var=49.0,
        ),
        histogram=histogram,
        track_window=selection,
        initialization_source=source,
    )


class MediaPipeCorrector:
    def __init__(self, config: TrackingConfig) -> None:
        self.mp = _import_mediapipe()
        self.hands = None
        if self.mp is not None:
            hands_cls = self._resolve_hands_class(self.mp)
            if hands_cls is not None:
                self.hands = hands_cls(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=config.mediapipe_min_detection_confidence,
                    min_tracking_confidence=config.mediapipe_min_tracking_confidence,
                )

    @staticmethod
    def _resolve_hands_class(mp: Any) -> Any | None:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            return solutions.hands.Hands
        try:
            from mediapipe.python.solutions.hands import Hands  # type: ignore
        except Exception:
            return None
        return Hands

    def detect(self, frame: Any) -> tuple[int, int, int, int] | None:
        if self.hands is None:
            return None
        rgb = frame[:, :, ::-1]
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        frame_height, frame_width = frame.shape[:2]
        landmarks = result.multi_hand_landmarks[0].landmark
        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        min_x = max(0, int(min(xs) * frame_width) - 18)
        max_x = min(frame_width - 1, int(max(xs) * frame_width) + 18)
        min_y = max(0, int(min(ys) * frame_height) - 18)
        max_y = min(frame_height - 1, int(max(ys) * frame_height) + 18)
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        return min_x, min_y, width, height

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()


def blend_histograms(cv2: Any, base_histogram: Any, new_histogram: Any, alpha: float) -> Any:
    blended = cv2.addWeighted(base_histogram, 1.0 - alpha, new_histogram, alpha, 0.0)
    cv2.normalize(blended, blended, 0, 255, cv2.NORM_MINMAX)
    return blended


def update_histogram_if_stable(
    cv2: Any,
    frame: Any,
    session: DemoSession,
    config: TrackingConfig,
) -> Any | None:
    if session.track_window is None or session.histogram is None:
        return None
    if session.last_confidence < config.histogram_update_threshold:
        return session.histogram
    if session.stable_accept_count < 3:
        return session.histogram
    fresh_histogram = create_histogram_model(cv2, frame, session.track_window)
    return blend_histograms(
        cv2=cv2,
        base_histogram=session.histogram,
        new_histogram=fresh_histogram,
        alpha=config.histogram_update_alpha,
    )


def try_auto_reinitialize(
    cv2: Any,
    frame: Any,
    histogram: Any,
    tracking_state: TrackingState,
    config: TrackingConfig,
) -> tuple[int, int, int, int] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 35, 35), (180, 255, 255))
    back_projection = cv2.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    back_projection = cv2.bitwise_and(back_projection, back_projection, mask=mask)
    back_projection = cv2.GaussianBlur(back_projection, (7, 7), 0)
    _, thresholded = cv2.threshold(
        back_projection,
        config.reinit_threshold_value,
        255,
        cv2.THRESH_BINARY,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    best_score = 0.0
    best_window: tuple[int, int, int, int] | None = None

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(width * height)
        if area < frame_area * config.reinit_area_ratio_min:
            continue
        if area > frame_area * config.reinit_area_ratio_max:
            continue
        roi = back_projection[y : y + height, x : x + width]
        if roi.size == 0:
            continue
        mean_score = float(roi.mean()) / 255.0
        candidate_cx = x + width / 2.0
        candidate_cy = y + height / 2.0
        dx = candidate_cx - tracking_state.cx
        dy = candidate_cy - tracking_state.cy
        diagonal = max((tracking_state.width * tracking_state.width + tracking_state.height * tracking_state.height) ** 0.5, 1.0)
        offset_penalty = min(1.0, ((dx * dx + dy * dy) ** 0.5) / (diagonal * 2.2))
        width_ratio = min(width, tracking_state.width) / max(width, tracking_state.width, 1.0)
        height_ratio = min(height, tracking_state.height) / max(height, tracking_state.height, 1.0)
        size_similarity = 0.5 * (width_ratio + height_ratio)
        score = mean_score * (1.0 - 0.55 * offset_penalty) * (0.65 + 0.35 * size_similarity)
        if score > best_score:
            best_score = score
            best_window = (x, y, width, height)

    if best_score < 0.18:
        return None
    return best_window


def apply_external_correction(
    cv2: Any,
    frame: Any,
    session: DemoSession,
    correction_window: tuple[int, int, int, int],
) -> DemoSession:
    corrected_session = initialize_session(
        cv2=cv2,
        frame=frame,
        selection=correction_window,
        source="mediapipe",
    )
    corrected_session.last_mediapipe_candidate = correction_window
    corrected_session.stable_accept_count = session.stable_accept_count
    corrected_session.frame_index = session.frame_index
    corrected_session.mediapipe_last_status = "hit"
    return corrected_session


def select_roi(cv2: Any, frame: Any, window_name: str) -> tuple[int, int, int, int] | None:
    selection = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    x, y, width, height = [int(value) for value in selection]
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def compute_measurement(
    cv2: Any,
    frame: Any,
    predicted_window: tuple[int, int, int, int],
    histogram: Any,
    tracking_state: TrackingState,
    config: TrackingConfig,
) -> tuple[Measurement, tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 35, 35), (180, 255, 255))
    back_projection = cv2.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    back_projection = cv2.bitwise_and(back_projection, back_projection, mask=mask)
    back_projection = cv2.GaussianBlur(back_projection, (5, 5), 0)
    frame_height, frame_width = back_projection.shape[:2]
    px, py, pwidth, pheight = predicted_window
    expanded_width = int(round(pwidth * config.local_search_expansion))
    expanded_height = int(round(pheight * config.local_search_expansion))
    expanded_x = max(0, px - (expanded_width - pwidth) // 2)
    expanded_y = max(0, py - (expanded_height - pheight) // 2)
    expanded_width = min(expanded_width, frame_width - expanded_x)
    expanded_height = min(expanded_height, frame_height - expanded_y)
    local_mask = back_projection.copy()
    local_mask[:, :] = 0
    local_mask[
        expanded_y : expanded_y + expanded_height,
        expanded_x : expanded_x + expanded_width,
    ] = 255
    back_projection = cv2.bitwise_and(back_projection, local_mask)
    spatial_prior = build_spatial_prior(
        frame_width=frame_width,
        frame_height=frame_height,
        center_x=tracking_state.cx,
        center_y=tracking_state.cy,
        sigma_x=max(tracking_state.width * 0.7, pwidth * 0.55),
        sigma_y=max(tracking_state.height * 0.7, pheight * 0.55),
    )
    weighted_projection = cv2.multiply(back_projection, spatial_prior, scale=1.0 / 255.0)
    _, thresholded = cv2.threshold(weighted_projection, 55, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    _, next_window = cv2.CamShift(
        cleaned,
        predicted_window,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1),
    )
    x, y, width, height = next_window
    roi_back_projection = weighted_projection[y : y + height, x : x + width]
    roi_cleaned = cleaned[y : y + height, x : x + width]
    confidence = 0.0
    if roi_back_projection.size > 0:
        mean_score = float(roi_back_projection.mean()) / 255.0
        peak_score = float(roi_back_projection.max()) / 255.0
        active_ratio = float((roi_cleaned > 0).mean()) if roi_cleaned.size > 0 else 0.0
        measurement_cx = x + width / 2.0
        measurement_cy = y + height / 2.0
        dx = measurement_cx - tracking_state.cx
        dy = measurement_cy - tracking_state.cy
        diagonal = max((tracking_state.width * tracking_state.width + tracking_state.height * tracking_state.height) ** 0.5, 1.0)
        normalized_offset = min(1.0, ((dx * dx + dy * dy) ** 0.5) / diagonal)
        width_ratio = min(width, tracking_state.width) / max(width, tracking_state.width, 1.0)
        height_ratio = min(height, tracking_state.height) / max(height, tracking_state.height, 1.0)
        size_similarity = 0.5 * (width_ratio + height_ratio)
        raw_confidence = 0.40 * mean_score + 0.25 * peak_score + 0.20 * active_ratio + 0.15 * size_similarity
        confidence = raw_confidence
        confidence *= max(0.0, 1.0 - config.center_penalty_factor * normalized_offset)
        confidence *= (1.0 - config.size_penalty_factor) + config.size_penalty_factor * size_similarity
        confidence = min(1.0, confidence)
    measurement = measurement_from_window(next_window, confidence)
    return measurement, next_window


def draw_overlay(
    cv2: Any,
    frame: Any,
    session: DemoSession,
    predicted_window: tuple[int, int, int, int] | None,
    fps: float,
) -> Any:
    output = frame.copy()
    if predicted_window is not None:
        x, y, width, height = predicted_window
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 180, 0), 1)
    if session.track_window is not None:
        x, y, width, height = session.track_window
        box_color = (0, 220, 0) if session.accepted_measurement else (0, 0, 255)
        cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 2)
    if session.last_mediapipe_candidate is not None:
        x, y, width, height = session.last_mediapipe_candidate
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    if session.last_reinit_candidate is not None:
        x, y, width, height = session.last_reinit_candidate
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 255), 2)
    mode_text = session.mode.value if session.mode is not None else "idle"
    lines = [
        f"Mode: {mode_text}",
        f"Init source: {session.initialization_source}",
        f"Lost count: {session.lost_count}",
        f"Accepted: {session.accepted_measurement}",
        f"Confidence: {session.last_confidence:.2f}",
        f"Distance^2: {session.last_distance:.2f}" if session.last_distance is not None else "Distance^2: -",
        f"Measurement weight: {session.last_blend_weight:.2f}",
        f"Stable accepts: {session.stable_accept_count}",
        f"Unstable frames: {session.unstable_frame_count}",
        f"MediaPipe: {session.mediapipe_last_status}",
        "Yellow: MediaPipe correction" ,
        "Purple: Color reinit candidate",
        f"FPS: {fps:.1f}",
        "Controls: r = manual ROI, q = quit",
    ]
    if session.mode == TrackMode.REINIT:
        lines.append("MediaPipe reinit active. Press r for manual ROI.")
    if session.mode is None:
        lines.append("Show your hand to auto-initialize with MediaPipe.")
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (16, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def run_demo(camera_index: int) -> None:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    window_name = "FoodNinja Tracking Demo"
    cv2.namedWindow(window_name)
    motion_model = KalmanMotionModel()
    mediapipe_corrector = MediaPipeCorrector(TrackingConfig())
    config = TrackingConfig(
        base_search_window=60.0,
        search_window_scale=1.2,
        gating_threshold=18.0,
        confidence_threshold=0.12,
        lost_reinit_threshold=12,
        measurement_blend_min=0.18,
        measurement_blend_max=0.68,
        speed_penalty_factor=0.0009,
        local_search_expansion=1.18,
        center_penalty_factor=0.75,
        size_penalty_factor=0.25,
        velocity_lookahead_factor=0.0,
        directional_search_gain=0.0,
        measurement_bonus_factor=0.0,
        fast_motion_threshold=220.0,
        bonus_distance_threshold=0.35,
        histogram_update_threshold=0.58,
        histogram_update_alpha=0.08,
        reinit_threshold_value=60,
        reinit_area_ratio_min=0.002,
        reinit_area_ratio_max=0.18,
        mediapipe_correction_interval_frames=10,
        mediapipe_low_confidence_threshold=0.55,
        mediapipe_min_detection_confidence=0.5,
        mediapipe_min_tracking_confidence=0.5,
        mediapipe_distance_trigger_ratio=0.45,
        mediapipe_unstable_frame_threshold=2,
    )
    session = DemoSession()
    previous_timestamp = time.perf_counter()
    fps = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Failed to read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt = max(1.0 / 120.0, now - previous_timestamp)
            previous_timestamp = now
            fps = 1.0 / dt
            session.frame_index += 1
            predicted_window = None

            if session.mode is None:
                session.mediapipe_last_status = "running"
                startup_window = mediapipe_corrector.detect(frame)
                session.last_mediapipe_candidate = startup_window
                if startup_window is not None:
                    session = initialize_session(
                        cv2=cv2,
                        frame=frame,
                        selection=startup_window,
                        source="mediapipe",
                    )
                    session.last_mediapipe_candidate = startup_window
                    session.mediapipe_last_status = "hit"
                else:
                    session.mediapipe_last_status = "miss"

            if (
                session.mode is not None
                and session.tracking_state is not None
                and session.covariance is not None
                and session.histogram is not None
            ):
                session.last_reinit_candidate = None
                session.last_mediapipe_candidate = None
                predicted_state, predicted_covariance = motion_model.predict(
                    state=session.tracking_state,
                    covariance=session.covariance,
                    dt=dt,
                )
                search_window = build_search_window(predicted_state, predicted_covariance, config)
                predicted_window = clamp_search_window(
                    search_window,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )

                measurement: Measurement | None = None
                next_window = session.track_window
                if predicted_window[2] > 0 and predicted_window[3] > 0:
                    measurement, next_window = compute_measurement(
                        cv2=cv2,
                        frame=frame,
                        predicted_window=predicted_window,
                        histogram=session.histogram,
                        tracking_state=session.tracking_state,
                        config=config,
                    )
                    session.last_confidence = measurement.confidence
                    session.last_distance = squared_mahalanobis_distance(
                        predicted_state=predicted_state,
                        measurement=measurement,
                        position_variance_x=predicted_covariance.pos_x_var,
                        position_variance_y=predicted_covariance.pos_y_var,
                    )
                else:
                    session.last_confidence = 0.0
                    session.last_distance = None

                result = advance_tracking_step(
                    current_mode=session.mode,
                    predicted_state=predicted_state,
                    measurement=measurement,
                    position_variance_x=predicted_covariance.pos_x_var,
                    position_variance_y=predicted_covariance.pos_y_var,
                    lost_count=session.lost_count,
                    config=config,
                )
                session.mode = result.mode
                session.lost_count = result.lost_count
                session.accepted_measurement = result.accepted_measurement

                if measurement is not None and result.accepted_measurement:
                    distance_squared = session.last_distance if session.last_distance is not None else 0.0
                    normalized_distance = min(1.0, distance_squared / max(config.gating_threshold, 1e-6))
                    base_weight = measurement.confidence * (1.0 - 0.55 * normalized_distance)
                    current_speed = (
                        session.tracking_state.vx * session.tracking_state.vx
                        + session.tracking_state.vy * session.tracking_state.vy
                    ) ** 0.5
                    speed_penalty = min(0.45, current_speed * config.speed_penalty_factor)
                    session.last_blend_weight = max(
                        config.measurement_blend_min,
                        min(config.measurement_blend_max, base_weight - speed_penalty),
                    )
                    session.tracking_state = blended_tracking_state(
                        predicted_state=predicted_state,
                        accepted_measurement=measurement,
                        dt=dt,
                        confidence=measurement.confidence,
                        distance_squared=distance_squared,
                        config=config,
                    )
                    session.stable_accept_count += 1
                    session.covariance = shrink_covariance_after_acceptance(predicted_covariance)
                    session.track_window = next_window
                    session.histogram = update_histogram_if_stable(
                        cv2=cv2,
                        frame=frame,
                        session=session,
                        config=config,
                    )
                    if (
                        session.last_distance is not None
                        and session.last_distance / max(config.gating_threshold, 1e-6) > config.mediapipe_distance_trigger_ratio
                    ) or session.last_confidence < config.mediapipe_low_confidence_threshold:
                        session.unstable_frame_count += 1
                    else:
                        session.unstable_frame_count = 0
                else:
                    session.last_blend_weight = 0.0
                    session.stable_accept_count = 0
                    session.unstable_frame_count += 1
                    session.tracking_state = predicted_state
                    session.covariance = grow_covariance_after_rejection(predicted_covariance)
                    session.track_window = predicted_window

                should_run_mediapipe = (
                    session.frame_index % config.mediapipe_correction_interval_frames == 0
                    or session.last_confidence <= config.mediapipe_low_confidence_threshold
                    or session.mode == TrackMode.REINIT
                    or session.unstable_frame_count >= config.mediapipe_unstable_frame_threshold
                    or (
                        session.last_distance is not None
                        and session.last_distance / max(config.gating_threshold, 1e-6) > config.mediapipe_distance_trigger_ratio
                    )
                )
                if should_run_mediapipe:
                    session.mediapipe_last_status = "running"
                    mediapipe_window = mediapipe_corrector.detect(frame)
                    session.last_mediapipe_candidate = mediapipe_window
                    if mediapipe_window is not None:
                        session = apply_external_correction(
                            cv2=cv2,
                            frame=frame,
                            session=session,
                            correction_window=mediapipe_window,
                        )
                        session.unstable_frame_count = 0
                    else:
                        session.mediapipe_last_status = "miss"
                else:
                    session.mediapipe_last_status = "idle"

                if session.mode == TrackMode.REINIT:
                    session.accepted_measurement = False
                    mediapipe_window = mediapipe_corrector.detect(frame)
                    session.last_mediapipe_candidate = mediapipe_window
                    if mediapipe_window is not None:
                        session = apply_external_correction(
                            cv2=cv2,
                            frame=frame,
                            session=session,
                            correction_window=mediapipe_window,
                        )
                        session.mediapipe_last_status = "hit"
                    else:
                        session.mediapipe_last_status = "miss"
                        auto_window = try_auto_reinitialize(
                            cv2=cv2,
                            frame=frame,
                            histogram=session.histogram,
                            tracking_state=session.tracking_state,
                            config=config,
                        )
                        session.last_reinit_candidate = auto_window
                        if auto_window is not None:
                            recovered_session = initialize_session(
                                cv2=cv2,
                                frame=frame,
                                selection=auto_window,
                                source="color-reinit",
                            )
                            recovered_session.last_reinit_candidate = auto_window
                            session = recovered_session

            display_frame = draw_overlay(
                cv2=cv2,
                frame=frame,
                session=session,
                predicted_window=predicted_window,
                fps=fps,
            )
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                selection = select_roi(cv2, frame, window_name)
                if selection is not None:
                    session = initialize_session(cv2=cv2, frame=frame, selection=selection, source="manual")
    finally:
        mediapipe_corrector.close()
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first FoodNinja hand-tracking demo.")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    run_demo(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
