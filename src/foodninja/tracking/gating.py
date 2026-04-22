from foodninja.config import TrackingConfig
from foodninja.core.models import Measurement, TrackingState


def squared_mahalanobis_distance(
    predicted_state: TrackingState,
    measurement: Measurement,
    position_variance_x: float,
    position_variance_y: float,
) -> float:
    dx = measurement.cx - predicted_state.cx
    dy = measurement.cy - predicted_state.cy
    safe_var_x = max(position_variance_x, 1e-6)
    safe_var_y = max(position_variance_y, 1e-6)
    return (dx * dx) / safe_var_x + (dy * dy) / safe_var_y


def accept_measurement(
    predicted_state: TrackingState,
    measurement: Measurement,
    position_variance_x: float,
    position_variance_y: float,
    config: TrackingConfig,
) -> bool:
    distance = squared_mahalanobis_distance(
        predicted_state=predicted_state,
        measurement=measurement,
        position_variance_x=position_variance_x,
        position_variance_y=position_variance_y,
    )
    return (
        distance < config.gating_threshold
        and measurement.confidence >= config.confidence_threshold
    )

