"""
Deprecated: lightweight scaffold predictor retained for demo/test compatibility.

The main game uses ``kalman_filter.py`` (full 6-state Kalman with numpy matrices).
This module is only imported by ``demo/tracking_demo.py``.
"""
from foodninja.core.models import CovarianceState, TrackingState


class KalmanMotionModel:
    """A lightweight constant-velocity predictor used as a project scaffold."""

    def predict(
        self,
        state: TrackingState,
        covariance: CovarianceState,
        dt: float,
    ) -> tuple[TrackingState, CovarianceState]:
        predicted_state = TrackingState(
            cx=state.cx + state.vx * dt,
            cy=state.cy + state.vy * dt,
            vx=state.vx,
            vy=state.vy,
            width=state.width,
            height=state.height,
        )
        predicted_covariance = CovarianceState(
            pos_x_var=covariance.pos_x_var + covariance.vel_x_var * dt,
            pos_y_var=covariance.pos_y_var + covariance.vel_y_var * dt,
            vel_x_var=covariance.vel_x_var,
            vel_y_var=covariance.vel_y_var,
            width_var=covariance.width_var,
            height_var=covariance.height_var,
        )
        return predicted_state, predicted_covariance

