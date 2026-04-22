from foodninja.core.models import CovarianceState, Measurement, SearchWindow, TrackingState
from foodninja.tracking.runtime import (
    clamp_search_window,
    corrected_tracking_state,
    grow_covariance_after_rejection,
    shrink_covariance_after_acceptance,
)


def test_clamp_search_window_stays_inside_frame() -> None:
    result = clamp_search_window(
        SearchWindow(x=-10.0, y=-20.0, width=100.0, height=80.0),
        frame_width=64,
        frame_height=48,
    )
    assert result == (0, 0, 64, 48)


def test_corrected_tracking_state_updates_velocity() -> None:
    previous_state = TrackingState(cx=10.0, cy=20.0, vx=0.0, vy=0.0, width=30.0, height=40.0)
    measurement = Measurement(cx=16.0, cy=32.0, width=28.0, height=38.0, confidence=0.9)
    corrected = corrected_tracking_state(previous_state, measurement, dt=0.5)
    assert corrected.vx == 12.0
    assert corrected.vy == 24.0


def test_covariance_adjustments_move_in_expected_direction() -> None:
    covariance = CovarianceState(100.0, 100.0, 200.0, 200.0, 50.0, 50.0)
    shrunk = shrink_covariance_after_acceptance(covariance)
    grown = grow_covariance_after_rejection(covariance)
    assert shrunk.pos_x_var < covariance.pos_x_var
    assert grown.pos_x_var > covariance.pos_x_var
