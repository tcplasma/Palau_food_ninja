from dataclasses import dataclass
from enum import Enum


class TrackMode(str, Enum):
    TRACKING = "tracking"
    LOST = "lost"
    REINIT = "reinit"


class FoodGroup(str, Enum):
    PROTECTIVE = "protective"
    ENERGY = "energy"
    BODY_BUILDING = "body_building"
    BOMB = "bomb"


@dataclass(slots=True)
class TrackingState:
    cx: float
    cy: float
    vx: float
    vy: float
    width: float
    height: float

    def to_array(self) -> "np.ndarray":
        import numpy as np
        return np.array([self.cx, self.cy, self.vx, self.vy, self.width, self.height], dtype=np.float32).reshape(6, 1)

    @classmethod
    def from_array(cls, arr: "np.ndarray"):
        arr = arr.flatten()
        return cls(cx=float(arr[0]), cy=float(arr[1]), vx=float(arr[2]), vy=float(arr[3]), width=float(arr[4]), height=float(arr[5]))

    def to_roi(self) -> tuple[int, int, int, int]:
        """Converts center/size to (x, y, w, h) ROI."""
        return (
            int(self.cx - self.width / 2),
            int(self.cy - self.height / 2),
            int(self.width),
            int(self.height)
        )


@dataclass(slots=True)
class CovarianceState:
    pos_x_var: float
    pos_y_var: float
    vel_x_var: float
    vel_y_var: float
    width_var: float
    height_var: float

    def trace(self) -> float:
        return (
            self.pos_x_var
            + self.pos_y_var
            + self.vel_x_var
            + self.vel_y_var
            + self.width_var
            + self.height_var
        )


@dataclass(slots=True)
class Measurement:
    cx: float
    cy: float
    width: float
    height: float
    confidence: float

    def to_array(self) -> "np.ndarray":
        import numpy as np
        return np.array([self.cx, self.cy, self.width, self.height], dtype=np.float32).reshape(4, 1)

    @classmethod
    def from_array(cls, arr: "np.ndarray", confidence: float = 1.0):
        arr = arr.flatten()
        return cls(cx=float(arr[0]), cy=float(arr[1]), width=float(arr[2]), height=float(arr[3]), confidence=confidence)


@dataclass(slots=True)
class SearchWindow:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class TrackingStepResult:
    mode: TrackMode
    state: TrackingState
    lost_count: int
    accepted_measurement: bool
    frames_since_reinit: int
    confidence: float
    trajectory: list[tuple[float, float]] # Last N positions for visual effect


@dataclass(slots=True)
class SpawnItem:
    name: str
    food_group: FoodGroup
    sprite_path: str
    start_x: float
    start_y: float
    velocity_x: float
    velocity_y: float
    angle: float = 0.0
    rotation_speed: float = 0.0

