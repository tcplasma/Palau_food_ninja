from dataclasses import dataclass


@dataclass(slots=True)
class TrackingConfig:
    base_search_window: float = 80.0
    search_window_scale: float = 2.5
    # Gating: P[pos] shrinks to ~9 after updates → sqrt(threshold * 9) = max allowed
    # pixels per frame. 150 → ~37px/frame at 60fps; handles ~2200px/sec hand motion.
    gating_threshold: float = 150.0
    confidence_threshold: float = 0.35
    lost_reinit_threshold: int = 45    # 1.5 second delay before full reinit
    measurement_blend_min: float = 0.20
    measurement_blend_max: float = 0.85
    speed_penalty_factor: float = 0.0009
    local_search_expansion: float = 2.5
    center_penalty_factor: float = 0.75
    size_penalty_factor: float = 0.25
    # Velocity lookahead: shifts CamShift search window along predicted trajectory.
    # 0.5 means search center moves by 0.5 * (vx*dt, vy*dt) ahead.
    velocity_lookahead_factor: float = 0.5
    directional_search_gain: float = 0.0
    measurement_bonus_factor: float = 0.0
    fast_motion_threshold: float = 220.0
    bonus_distance_threshold: float = 0.35
    histogram_update_threshold: float = 0.58
    histogram_update_alpha: float = 0.08
    reinit_threshold_value: int = 60
    reinit_area_ratio_min: float = 0.002
    reinit_area_ratio_max: float = 0.18
    # MediaPipe correction: every 3 frames keeps ground-truth anchoring frequent.
    # The background thread is non-blocking so this doesn't affect frame rate.
    mediapipe_correction_interval_frames: int = 3
    mediapipe_low_confidence_threshold: float = 0.45
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    mediapipe_distance_trigger_ratio: float = 0.45
    mediapipe_unstable_frame_threshold: int = 2
    mediapipe_model_path: str = "assets/models/hand_landmarker.task"
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.05
    trajectory_history_length: int = 15


@dataclass(slots=True)
class SpawnConfig:
    screen_width: int = 1280
    screen_height: int = 720
    min_launch_speed_y: float = 680.0
    max_launch_speed_y: float = 920.0
    min_launch_speed_x: float = -260.0
    max_launch_speed_x: float = 260.0
    gravity: float = 980.0
    hazard_chance: float = 0.2 # 20% chance per wave to include a trap
    max_hazards_per_wave: int = 1
