from foodninja.config import TrackingConfig
from foodninja.core.models import CovarianceState, Measurement, SearchWindow, TrackingState


def clamp_search_window(window: SearchWindow, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x = max(0, int(round(window.x)))
    y = max(0, int(round(window.y)))
    width = max(1, int(round(window.width)))
    height = max(1, int(round(window.height)))
    if x + width > frame_width:
        width = max(1, frame_width - x)
    if y + height > frame_height:
        height = max(1, frame_height - y)
    return x, y, width, height


def measurement_from_window(window: tuple[int, int, int, int], confidence: float) -> Measurement:
    x, y, width, height = window
    return Measurement(
        cx=x + width / 2.0,
        cy=y + height / 2.0,
        width=float(width),
        height=float(height),
        confidence=confidence,
    )


def corrected_tracking_state(
    previous_state: TrackingState,
    accepted_measurement: Measurement,
    dt: float,
) -> TrackingState:
    safe_dt = max(dt, 1e-6)
    vx = (accepted_measurement.cx - previous_state.cx) / safe_dt
    vy = (accepted_measurement.cy - previous_state.cy) / safe_dt
    return TrackingState(
        cx=accepted_measurement.cx,
        cy=accepted_measurement.cy,
        vx=vx,
        vy=vy,
        width=accepted_measurement.width,
        height=accepted_measurement.height,
    )


def blended_tracking_state(
    predicted_state: TrackingState,
    accepted_measurement: Measurement,
    dt: float,
    confidence: float,
    distance_squared: float,
    config: TrackingConfig,
) -> TrackingState:
    safe_dt = max(dt, 1e-6)
    speed = (predicted_state.vx * predicted_state.vx + predicted_state.vy * predicted_state.vy) ** 0.5
    normalized_distance = min(1.0, distance_squared / max(config.gating_threshold, 1e-6))
    base_weight = confidence * (1.0 - 0.55 * normalized_distance)
    speed_penalty = min(0.45, speed * config.speed_penalty_factor)
    measurement_weight = max(
        config.measurement_blend_min,
        min(config.measurement_blend_max, base_weight - speed_penalty),
    )
    prediction_weight = 1.0 - measurement_weight

    blended_cx = prediction_weight * predicted_state.cx + measurement_weight * accepted_measurement.cx
    blended_cy = prediction_weight * predicted_state.cy + measurement_weight * accepted_measurement.cy
    blended_width = prediction_weight * predicted_state.width + measurement_weight * accepted_measurement.width
    blended_height = prediction_weight * predicted_state.height + measurement_weight * accepted_measurement.height
    blended_vx = 0.72 * predicted_state.vx + (blended_cx - predicted_state.cx) / safe_dt
    blended_vy = 0.72 * predicted_state.vy + (blended_cy - predicted_state.cy) / safe_dt

    return TrackingState(
        cx=blended_cx,
        cy=blended_cy,
        vx=blended_vx,
        vy=blended_vy,
        width=blended_width,
        height=blended_height,
    )


def shrink_covariance_after_acceptance(covariance: CovarianceState) -> CovarianceState:
    return CovarianceState(
        pos_x_var=max(9.0, covariance.pos_x_var * 0.55),
        pos_y_var=max(9.0, covariance.pos_y_var * 0.55),
        vel_x_var=max(25.0, covariance.vel_x_var * 0.8),
        vel_y_var=max(25.0, covariance.vel_y_var * 0.8),
        width_var=max(16.0, covariance.width_var * 0.7),
        height_var=max(16.0, covariance.height_var * 0.7),
    )


def grow_covariance_after_rejection(covariance: CovarianceState) -> CovarianceState:
    return CovarianceState(
        pos_x_var=min(1200.0, covariance.pos_x_var * 1.35),
        pos_y_var=min(1200.0, covariance.pos_y_var * 1.35),
        vel_x_var=min(2000.0, covariance.vel_x_var * 1.2),
        vel_y_var=min(2000.0, covariance.vel_y_var * 1.2),
        width_var=min(800.0, covariance.width_var * 1.15),
        height_var=min(800.0, covariance.height_var * 1.15),
    )
