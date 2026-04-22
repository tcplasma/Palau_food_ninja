from foodninja.config import TrackingConfig
from foodninja.core.models import Measurement, TrackMode, TrackingState, TrackingStepResult
from foodninja.tracking.gating import accept_measurement


def resolve_track_mode(
    current_mode: TrackMode,
    measurement_accepted: bool,
    lost_count: int,
    frames_since_reinit: int,
    config: TrackingConfig,
) -> TrackMode:
    if frames_since_reinit >= config.mediapipe_correction_interval_frames:
        return TrackMode.REINIT
    if measurement_accepted:
        return TrackMode.TRACKING
    if lost_count > config.lost_reinit_threshold:
        return TrackMode.REINIT
    if current_mode == TrackMode.REINIT:
        return TrackMode.REINIT
    return TrackMode.LOST


def advance_tracking_step(
    current_mode: TrackMode,
    predicted_state: TrackingState,
    measurement: Measurement | None,
    position_variance_x: float,
    position_variance_y: float,
    lost_count: int,
    frames_since_reinit: int,
    config: TrackingConfig,
) -> TrackingStepResult:
    measurement_accepted = False
    next_state = predicted_state

    if measurement is not None:
        measurement_accepted = accept_measurement(
            predicted_state=predicted_state,
            measurement=measurement,
            position_variance_x=position_variance_x,
            position_variance_y=position_variance_y,
            config=config,
        )
        if measurement_accepted:
            next_state = TrackingState(
                cx=measurement.cx,
                cy=measurement.cy,
                vx=predicted_state.vx,
                vy=predicted_state.vy,
                width=measurement.width,
                height=measurement.height,
            )

    next_lost_count = 0 if measurement_accepted else lost_count + 1
    next_frames_since_reinit = frames_since_reinit + 1
    next_mode = resolve_track_mode(
        current_mode=current_mode,
        measurement_accepted=measurement_accepted,
        lost_count=next_lost_count,
        frames_since_reinit=next_frames_since_reinit,
        config=config,
    )
    if next_mode == TrackMode.REINIT:
        next_frames_since_reinit = 0

    return TrackingStepResult(
        mode=next_mode,
        state=next_state,
        lost_count=next_lost_count,
        accepted_measurement=measurement_accepted,
        frames_since_reinit=next_frames_since_reinit,
    )

