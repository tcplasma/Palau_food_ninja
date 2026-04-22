# Edge-Based Hand Tracking Nutrition Game

An English-first Python project for a low-latency nutrition education game inspired by Fruit Ninja.

## Project Goals

- Run on low-power edge devices such as Raspberry Pi 5 or CPU-only PCs
- Avoid continuous deep learning inference during normal tracking
- Maintain responsive tracking at 30 FPS or above
- Teach the three Palau food groups through gesture-based gameplay
- Package the final build as a Windows executable

## Architecture Layers

1. `camera`
   Capture frames from a webcam or video source.
2. `initialization`
   Detect and recover the hand when tracking is lost.
3. `tracking`
   Predict with Kalman, measure with CamShift, and gate with Mahalanobis distance.
4. `gesture`
   Interpret trajectories into game actions such as swipe and hit.
5. `gameplay`
   Spawn foods, score interactions, and drive the nutrition-learning loop.

## Project Layout

```text
assets/
  food_groups/
docs/
src/
  foodninja/
tests/
```

## Current Status

This repository currently contains:

- a project skeleton aligned with the tracking architecture
- an expandable food asset taxonomy
- initial gameplay and tracking domain models
- starter tests for gating, state transitions, and spawn behavior

## Next Steps

1. Install Python 3.11 or newer on the development machine.
2. Install dependencies for OpenCV and testing.
3. Implement frame acquisition, histogram initialization, and live tracking integration.
4. Add synthetic replay tests and live-camera validation.
5. Package with PyInstaller once the first playable loop is stable.

## Run The First Demo

Install the vision dependencies first:

```powershell
pip install -e .[dev,vision]
```

Then start the webcam demo:

```powershell
python -m foodninja.demo.tracking_demo
```

Demo controls:

- press `r` to select the initial hand ROI
- press `q` to quit

The demo overlays:

- current tracking mode
- lost counter
- measurement acceptance
- confidence score
- FPS
