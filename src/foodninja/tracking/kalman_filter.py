import numpy as np
from typing import Tuple

class KalmanFilterModule:
    """
    Kalman Filter for predicting and updating bounded boxes in the tracking state.
    State x = [cx, cy, vx, vy, w, h]^T
    Measurement z = [cx, cy, w, h]^T
    """
    def __init__(self, dt: float = 1.0/60.0):
        # State transition matrix (Constant Velocity Model)
        self._dt = dt
        self.F = self._make_F(dt)

        # Measurement matrix (Maps state to measurement)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Process noise covariance Q.
        # Position: small (Kalman smooths it). Velocity: large so fast hand
        # movements don't starve the gating window. Size: moderate.
        self.Q = np.diag([1.0, 1.0, 800.0, 800.0, 4.0, 4.0]).astype(np.float32)

        # Measurement noise covariance R
        self.R = np.eye(4, dtype=np.float32) * 40.0

        # State and Covariance initialization
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

    @staticmethod
    def _make_F(dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1,  0, 0, 0],
            [0, 0, 0,  1, 0, 0],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1]
        ], dtype=np.float32)

    def update_dt(self, dt: float) -> None:
        """Update F with the actual measured frame dt for accurate prediction."""
        if abs(dt - self._dt) > 1e-4:
            self._dt = dt
            self.F = self._make_F(dt)

    def initialize(self, initial_state: np.ndarray):
        """Reinitialize state x and covariance P."""
        self.x = initial_state.reshape(6, 1)
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the state with a new measurement."""
        z = z.reshape(4, 1)
        y = z - (self.H @ self.x) # Innovation
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        self.x = self.x + (K @ y)
        I = np.eye(self.P.shape[0])
        self.P = (I - (K @ self.H)) @ self.P
        return self.x.copy(), self.P.copy()
