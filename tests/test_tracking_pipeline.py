from foodninja.config import TrackingConfig
from foodninja.core.models import Measurement, TrackMode, TrackingState
from foodninja.tracking.pipeline import advance_tracking_step


def test_tracking_pipeline_enters_lost_after_rejection() -> None:
    config = TrackingConfig(lost_reinit_threshold=3, confidence_threshold=0.8)
    state = TrackingState(cx=100.0, cy=100.0, vx=0.0, vy=0.0, width=50.0, height=50.0)
    measurement = Measurement(cx=250.0, cy=250.0, width=50.0, height=50.0, confidence=0.2)
    result = advance_tracking_step(
        current_mode=TrackMode.TRACKING,
        predicted_state=state,
        measurement=measurement,
        position_variance_x=4.0,
        position_variance_y=4.0,
        lost_count=0,
        config=config,
    )
    assert result.mode == TrackMode.LOST
    assert result.lost_count == 1


def test_tracking_pipeline_enters_reinit_after_repeated_loss() -> None:
    config = TrackingConfig(lost_reinit_threshold=2)
    state = TrackingState(cx=100.0, cy=100.0, vx=0.0, vy=0.0, width=50.0, height=50.0)
    result = advance_tracking_step(
        current_mode=TrackMode.LOST,
        predicted_state=state,
        measurement=None,
        position_variance_x=4.0,
        position_variance_y=4.0,
        lost_count=2,
        config=config,
    )
    assert result.mode == TrackMode.REINIT
