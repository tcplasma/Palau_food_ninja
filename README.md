# 🥷 Food Ninja — Edge-Based Hand Tracking Nutrition Game

A low-latency nutrition education game inspired by Fruit Ninja.
Players use their bare hand (tracked via webcam) to slash flying food items and learn the three
Palau food groups: **Protective**, **Energy**, and **Body Building**.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Architecture Overview](#architecture-overview)
3. [Module Reference](#module-reference)
4. [Game Loop Walkthrough](#game-loop-walkthrough)
5. [Setup & Running](#setup--running)
6. [Project Layout](#project-layout)

---

## Project Goals

| Goal | Detail |
|------|--------|
| **Edge-first** | Run on low-power devices (RPi 5, CPU-only PC) at ≥30 FPS |
| **Hybrid tracking** | Use MediaPipe only for init/recovery; CamShift + Kalman for real-time |
| **Nutrition education** | Teach Palau's three food groups through timed slash challenges |
| **Distributable** | Package as a single Windows `.exe` via PyInstaller |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    app.py (Coordination Layer)               │
│     Camera → HandTracker → GameState → GameRenderer          │
└────┬──────────┬───────────────┬───────────────┬──────────────┘
     │          │               │               │
     ▼          ▼               ▼               ▼
  camera/   tracking/       gameplay/       gameplay/
  source    tracker       game_state.py    renderer.py
              │               │
     ┌────────┼────────┐      ├── spawner
     ▼        ▼        ▼      ├── slice_engine
  kalman_  camshift  filters  ├── session (ScoreBoard)
  filter              │      ├── food_catalog
     │                │      └── audio_engine
     │           gating.py
     │
  initialization/
  mediapipe_recovery
```

### Layer Summary

| Layer | Role |
|-------|------|
| `app.py` | Thin coordination layer (~115 lines): camera → tracker → game_state → renderer |
| `camera/` | Webcam capture abstraction (`CameraSource`, `CameraFrame`) |
| `initialization/` | MediaPipe Tasks hand landmarker for first-detect & recovery |
| `tracking/` | Kalman predict → CamShift measure → Mahalanobis gating → Kalman update |
| `gesture/` | Trajectory → game action mapping (swipe/hit speed thresholds) — scaffold |
| `gameplay/` | Game state, spawner, slice collision, scoring, rendering, audio, catalogs |
| `core/` | Shared data models and `get_resource_path` utility |

---

## Module Reference

### `app.py` — Entry Point & Coordination

Wires together camera capture, `HandTracker`, `GameState`, and `GameRenderer`.
The main loop is:

1. Event handling (quit / 'Q')
2. Camera read → tracker.process_frame()
3. game_state.update() → process_slices() → check_game_over()
4. renderer.render_frame()

---

### `core/models.py` — Domain Data Models

| Class | Fields | Purpose |
|-------|--------|---------|
| `TrackMode` | `TRACKING`, `LOST`, `REINIT` | State machine modes for the tracker |
| `FoodGroup` | `PROTECTIVE`, `ENERGY`, `BODY_BUILDING`, `BOMB` | Food classification enum (BOMB = hazard) |
| `TrackingState` | cx, cy, vx, vy, width, height | Kalman filter state vector |
| `Measurement` | cx, cy, width, height, confidence | Raw observation from CamShift or MediaPipe |
| `SpawnItem` | name, food_group, sprite_path, start_x/y, velocity_x/y, angle, rotation_speed | A single flying food/bomb on screen |
| `TrackingStepResult` | mode, state, lost_count, accepted_measurement, confidence, trajectory | Per-frame output of the tracker |

### `core/utils.py` — Resource Path Resolution

`get_resource_path()` resolves asset paths for both dev (`os.getcwd()`) and PyInstaller (`sys._MEIPASS`).

---

### `tracking/tracker.py` — HandTracker Orchestrator

Ties all tracking components in a single `process_frame(frame, dt)` call:

1. Check background MediaPipe thread for ground-truth corrections
2. **Conditional handling**: during `TRACKING` → soft Kalman update; during `REINIT/LOST` → full reinit
3. Enqueue new frame to MediaPipe every N frames
4. Kalman predict → CamShift measure → Gating → Kalman update
5. One Euro smoothing → `visual_state`
6. Append to trajectory history (cyan trail effect)

### `tracking/kalman_filter.py` — 6-State Kalman Filter

- **State vector**: `[cx, cy, vx, vy, w, h]` (constant-velocity model)
- `update_dt(dt)` — rebuilds the transition matrix `F` for frame-rate independence
- **Process noise Q**: high velocity variance (800) for fast hand motion

### `tracking/camshift.py` — CamShift + Multi-Metric Confidence

1. **H-S 2D Histogram** — skin-colour filtered
2. **Backprojection** — Gaussian blur → morphological open/close → threshold
3. **Dynamic search window** — sized by Kalman uncertainty, shifted along velocity
4. **Multi-metric confidence** — mean (0.3) + peak (0.4) + active-pixel ratio (0.3) × spatial penalty
5. **EMA histogram update** — adapts to lighting changes at α = 0.05
6. `build_search_window()` — standalone function for the tracking demo

### `tracking/gating.py` — Mahalanobis Distance Gate

Accepts a CamShift measurement only when:
- Squared Mahalanobis distance < `gating_threshold` (150)
- Confidence ≥ `confidence_threshold` (0.35)

### `tracking/filters.py` — One Euro Filter

Smooths visual output with adaptive cutoff: rises with speed so fast slashes stay responsive.

### `tracking/kalman.py` — Scaffold Predictor (Deprecated)

Lightweight constant-velocity predictor. **Deprecated** — retained only for `demo/tracking_demo.py` compatibility. The main game uses `kalman_filter.py`.

### `tracking/pipeline.py` & `tracking/runtime.py` — Pure-Function Helpers

Stateless functions for state-machine transitions, blending, covariance adjustment. Used by tests and the tracking demo.

---

### `initialization/mediapipe_recovery.py` — MediaPipe Hand Detection

| Class | Purpose |
|-------|---------|
| `HandDetector` | Synchronous MediaPipe Tasks `HandLandmarker`. Computes bounding box from 21 landmarks. |
| `ThreadedHandDetector` | Daemon thread with size-1 queue. Non-blocking `enqueue_frame()` / `get_latest_result()`. |

---

### `gameplay/game_state.py` — Game Logic

`GameState` owns all mutable gameplay state and subsystems:

| Method | Responsibility |
|--------|---------------|
| `update(dt, tracking_result)` | Calibration handshake, target rotation, food/bomb spawning, physics |
| `process_slices(tracking_result)` | Slice detection → unified scoring via `ScoreBoard` |
| `check_game_over()` | Sets game-over when wrong_hits ≥ 10 or time > 120s |

**Scoring is single-entry-point**: `process_slices()` dispatches to `ScoreBoard.register_hit()` for food and `ScoreBoard.register_bomb_hit()` for bombs.

### `gameplay/renderer.py` — Rendering

`GameRenderer` draws every visual element — never mutates game state:

- Camera feed as background
- Cyan meteor trail (trajectory history)
- Rotated food/bomb sprites (80×80)
- Calibration zone + progress bar
- HUD (correct/wrong counts, target group)
- Game-over overlay with summary

### `gameplay/session.py` — ScoreBoard

| Method | Behaviour |
|--------|-----------|
| `register_hit(group)` | Correct target → `correct_hits += 1`; wrong group → `wrong_hits += 1` |
| `register_bomb_hit(penalty)` | `wrong_hits += penalty` (default 3, read from trap catalog) |

### `gameplay/slice_engine.py` — Collision Detection

- **Line-segment** collision: point-to-segment distance for robust high-speed detection
- **Hit radius**: `max(45px, hand_width / 2)` — adaptive to tracked bounding box
- **REINIT guard**: clears trajectory when tracking is unreliable
- **LOST passthrough**: Kalman prediction valid for ~10–15 frames, so slices allowed

### `gameplay/spawner.py` — FoodSpawner

Creates `SpawnItem` with randomised position, velocity, rotation.

### `gameplay/food_catalog.py` — Catalog Loader

Loads `assets/food_catalog.json` → `list[FoodDefinition]`. 35 items across three groups.

### `gameplay/audio_engine.py` — Voice & SFX

| Class | Detail |
|-------|--------|
| `VoiceAnnouncer` | Background-threaded TTS via `pyttsx3`. Caches WAVs to disk. |
| `SfxEngine` | Synthetic sine-sweep slice sounds via NumPy + `pygame.sndarray`. |

---

### `config.py` — Tuning Parameters

| Dataclass | Key Parameters |
|-----------|---------------|
| `TrackingConfig` | gating_threshold (150), confidence_threshold (0.35), lost_reinit_threshold (45), mediapipe_correction_interval (3), one_euro_beta (0.05) |
| `SpawnConfig` | screen 1280×720, gravity (980 px/s²), hazard_chance (20%) |

---

## Game Loop Walkthrough

```
app.py  FoodNinjaGame.run()
│
├─ 1. Event Handling (Pygame quit / 'Q' key)
│
├─ 2. Camera & Tracking
│   ├─ clock.tick(60) → dt
│   ├─ cap.read() → frame (flipped horizontally)
│   └─ tracker.process_frame(frame, dt) → TrackingStepResult
│
├─ 3. Game Logic (GameState)
│   ├─ update(dt, tracking_result)
│   │   ├─ Calibration: hold hand in centre 300×300 zone for 1.5s
│   │   ├─ Target rotation: every 15s (PROTECTIVE → ENERGY → BODY_BUILDING)
│   │   ├─ Spawning: every 2s, 1-4 foods + 20% chance of bomb
│   │   └─ Physics: gravity, position, rotation, off-screen cleanup
│   ├─ process_slices(tracking_result)
│   │   ├─ BOMB → sfx.play_wrong() + scoreboard.register_bomb_hit(penalty)
│   │   ├─ Correct group → sfx.play_correct() + scoreboard.register_hit()
│   │   └─ Wrong group → sfx.play_wrong() + scoreboard.register_hit()
│   └─ check_game_over() → wrong_hits ≥ 10 or time > 120s
│
└─ 4. Render (GameRenderer)
    ├─ Camera feed background
    ├─ Cyan meteor trail
    ├─ Food/bomb sprites
    ├─ HUD overlay
    └─ Game over summary
```

### Data-Driven Trap System

Bombs are defined in `assets/trap_catalog.json`:
```json
[{ "name": "BOMB", "effect": "instant_penalty", "penalty": 3, "sprite_path": "assets/food/bomb.png" }]
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Webcam

### Install

```powershell
pip install -e .[dev,vision]
pip install pyttsx3 pygame
```

### Run the Game

```powershell
python -m foodninja.app
```

### Run the Tracking Demo

```powershell
python -m foodninja.demo.tracking_demo
```

### Run Tests

```powershell
pytest
```

### Build Executable

```powershell
pip install -e .[packaging]
pyinstaller NutritionNinja.spec
```

---

## Project Layout

```
foodninja/
├── assets/
│   ├── audio/                     # Cached TTS WAV files
│   ├── food/                      # Food sprite PNGs (by group)
│   │   ├── protective/
│   │   ├── energy/
│   │   ├── body_building/
│   │   └── bomb.png
│   ├── food_groups/
│   ├── models/
│   │   └── hand_landmarker.task   # MediaPipe model
│   ├── food_catalog.json          # 35 food definitions
│   └── trap_catalog.json          # Hazard definitions
├── docs/
├── src/foodninja/
│   ├── app.py                     # Thin coordination layer (~115 lines)
│   ├── config.py                  # TrackingConfig & SpawnConfig
│   ├── camera/
│   │   └── source.py              # Camera abstraction (scaffold)
│   ├── core/
│   │   ├── models.py              # All domain data classes
│   │   └── utils.py               # get_resource_path()
│   ├── demo/
│   │   └── tracking_demo.py       # Standalone tracking visualiser
│   ├── gameplay/
│   │   ├── audio_engine.py        # VoiceAnnouncer + SfxEngine
│   │   ├── food_catalog.py        # JSON → FoodDefinition loader
│   │   ├── game_state.py          # [NEW] All game logic
│   │   ├── renderer.py            # [NEW] All Pygame rendering
│   │   ├── session.py             # ScoreBoard (single scoring entry point)
│   │   ├── slice_engine.py        # Line-segment collision detection
│   │   └── spawner.py             # Random food/bomb spawning
│   ├── gesture/
│   │   └── interpreter.py         # Speed → swipe/hit classifier (scaffold)
│   ├── initialization/
│   │   ├── interfaces.py          # InitializationResult
│   │   └── mediapipe_recovery.py  # HandDetector + ThreadedHandDetector
│   └── tracking/
│       ├── camshift.py            # CamShift + build_search_window()
│       ├── filters.py             # One Euro Filter
│       ├── gating.py              # Mahalanobis distance gate
│       ├── kalman.py              # Deprecated scaffold predictor (demo only)
│       ├── kalman_filter.py       # Full 6-state Kalman filter
│       ├── pipeline.py            # Pure-function state machine (tests/demo)
│       ├── runtime.py             # Blending/covariance helpers (tests/demo)
│       └── tracker.py             # Main HandTracker orchestrator
├── tests/
│   ├── test_food_catalog.py
│   ├── test_gating.py
│   ├── test_spawner.py
│   ├── test_tracking_pipeline.py
│   └── test_tracking_runtime.py
├── NutritionNinja.spec
├── pyproject.toml
└── README.md
```
