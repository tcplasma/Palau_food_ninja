import numpy as np
from typing import Tuple

class KalmanFilterModule:
    """
    Kalman Filter for predicting and updating bounded boxes in the tracking state.
    State x = [cx, cy, vx, vy, w, h]^T
    Measurement z = [cx, cy, w, h]^T
    """
    def __init__(self, dt: float = 1.0/30.0):
        # State transition matrix (Constant Velocity Model)
        self.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1,  0, 0, 0],
            [0, 0, 0,  1, 0, 0],
            [0, 0, 0,  0, 1, 0],
            [0, 0, 0,  0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (Maps state to measurement)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Process noise covariance Q (Reduced for higher trust in motion model)
        self.Q = np.eye(6, dtype=np.float32) * 0.01 
        self.Q[2:4, 2:4] *= 2.0 # Slightly higher uncertainty in velocity

        # Measurement noise covariance R (Balanced for 10-frame high-frequency logic)
        self.R = np.eye(4, dtype=np.float32) * 40.0

        # State and Covariance initialization
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

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
