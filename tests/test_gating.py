from foodninja.config import TrackingConfig
from foodninja.core.models import Measurement, TrackingState
from foodninja.tracking.gating import accept_measurement, squared_mahalanobis_distance


def test_squared_mahalanobis_distance_is_zero_for_perfect_measurement() -> None:
    state = TrackingState(cx=100.0, cy=150.0, vx=0.0, vy=0.0, width=40.0, height=50.0)
    measurement = Measurement(cx=100.0, cy=150.0, width=40.0, height=50.0, confidence=0.9)
    distance = squared_mahalanobis_distance(state, measurement, 4.0, 9.0)
    assert distance == 0.0


def test_accept_measurement_rejects_low_confidence() -> None:
    config = TrackingConfig(confidence_threshold=0.8)
    state = TrackingState(cx=100.0, cy=150.0, vx=0.0, vy=0.0, width=40.0, height=50.0)
    measurement = Measurement(cx=101.0, cy=149.0, width=42.0, height=52.0, confidence=0.6)
    assert not accept_measurement(state, measurement, 9.0, 9.0, config)

