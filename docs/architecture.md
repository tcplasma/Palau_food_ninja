# System Architecture

## Layered Modules

```text
Application Layer
  Nutrition Game / UI / UX

Gesture Interpretation
  Trajectory / Swipe / Hit

Tracking Core
  CamShift (measurement)
  Kalman Filter (prediction)
  Gating (Mahalanobis)

Initialization / Recovery
  MediaPipe or detector-based hand recovery

Camera Input
  Webcam frame source
```

## State Definition

The tracking state uses:

```text
x = [cx, cy, vx, vy, w, h]
```

- `cx`, `cy`: hand center
- `vx`, `vy`: velocity
- `w`, `h`: tracked bounding box size

## Frame Pipeline

```text
Frame_t
  -> Kalman predict
  -> predicted ROI
  -> CamShift measure
  -> gate measurement
  -> update or reject
  -> state output
  -> gesture interpretation
  -> game update
```

## Tracking Modes

- `TRACKING`
- `LOST`
- `REINIT`

## Recovery Rules

- Move from `TRACKING` to `LOST` when gating repeatedly fails.
- Move from `LOST` to `TRACKING` when a valid measurement is accepted.
- Move from `LOST` to `REINIT` when the lost counter exceeds the threshold.

## Edge Constraints

- Run Kalman and CamShift every frame.
- Run the detector only when needed for recovery.
- Limit processing to the predicted ROI whenever possible.
- Keep the tracking core independent from the UI and game rules.

## SOLID-Oriented Design Targets

- Single responsibility:
  - one class for prediction
  - one class for measurement
  - one class for gating
  - one class for mode transition logic
- Open for extension:
  - future upgrades can swap CamShift for hybrid tracking without changing gameplay code
- Dependency inversion:
  - the gameplay layer depends on tracking interfaces, not direct OpenCV calls
