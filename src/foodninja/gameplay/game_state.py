"""Game state management: calibration, spawning, physics, slicing, and scoring."""

import time
from typing import List

from foodninja.config import SpawnConfig
from foodninja.core.models import TrackMode, FoodGroup, SpawnItem, TrackingStepResult
from foodninja.gameplay.spawner import FoodSpawner
from foodninja.gameplay.slice_engine import SliceEngine
from foodninja.gameplay.session import ScoreBoard
from foodninja.gameplay.food_catalog import FoodDefinition
from foodninja.gameplay.audio_engine import VoiceAnnouncer, SfxEngine


class GameState:
    """
    Owns all mutable gameplay state and subsystems.

    Responsibilities:
    - Calibration handshake (hand-in-zone countdown)
    - Target group rotation on a timer
    - Food / hazard spawning waves
    - Per-frame physics (gravity, position, rotation, off-screen cleanup)
    - Slice detection → unified scoring via ScoreBoard
    - Game-over condition checks
    """

    def __init__(
        self,
        spawn_config: SpawnConfig,
        catalog: list[FoodDefinition],
        trap_catalog: list[dict],
    ):
        self.spawn_config = spawn_config
        self.catalog = catalog
        self.trap_catalog = trap_catalog

        # Subsystems
        self.spawner = FoodSpawner(spawn_config)
        self.slice_engine = SliceEngine()
        self.scoreboard = ScoreBoard()
        self.voice = VoiceAnnouncer()
        self.sfx = SfxEngine()

        # Active flying items
        self.active_items: List[SpawnItem] = []

        # Timers
        self.last_spawn_time = time.time()
        self.last_target_rotation_time = time.time()
        self.game_start_time = time.time()
        self.spawn_interval = 2.0
        self.target_rotation_interval = 15.0
        self.game_duration = 60.0

        # Flags
        self.is_game_over = False
        self.has_initialized = False
        self.calibration_progress = 0.0
        self.calibration_target_time = 1.5

        # Bomb visual feedback (countdown in seconds, >0 = flash active)
        self.bomb_flash_timer = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float, tracking_result: TrackingStepResult) -> None:
        """Advance game logic by one frame (calibration, spawning, physics)."""
        if self.is_game_over:
            return

        hand_state = tracking_result.state

        # 1. Calibration phase
        if not self.has_initialized:
            self._handle_calibration(dt, hand_state)
            return

        # 2. Rotate target group (skip BOMB)
        self._maybe_rotate_target()

        # 3. Spawn food + hazard roll
        self._maybe_spawn()

        # 4. Physics
        self._update_physics(dt)

    def process_slices(self, tracking_result: TrackingStepResult) -> None:
        """Detect slices and apply scoring — single entry point for all score logic."""
        if not self.has_initialized or self.is_game_over:
            return

        # Tick down bomb flash
        if self.bomb_flash_timer > 0:
            self.bomb_flash_timer -= 1.0 / 60.0  # approximate, good enough for flash

        slices = self.slice_engine.process_frame(
            tracking_result.state,
            self.active_items,
            tracking_mode=tracking_result.mode,
        )

        for slice_res in slices:
            if slice_res.item.food_group == FoodGroup.BOMB:
                self.sfx.play_bomb()
                trap_def = next(
                    (t for t in self.trap_catalog if t["name"] == slice_res.item.name),
                    None,
                )
                penalty = trap_def.get("penalty", 3) if trap_def else 3
                self.scoreboard.register_bomb_hit(penalty)
                self.bomb_flash_timer = 0.5  # 0.5 second red flash
                self.voice.announce(f"Bomb! Minus {penalty} points!")
            elif slice_res.item.food_group == self.scoreboard.current_target_group:
                self.sfx.play_correct()
                self.scoreboard.register_hit(slice_res.item.food_group)
            else:
                self.sfx.play_wrong()
                self.scoreboard.register_hit(slice_res.item.food_group)

            if slice_res.item in self.active_items:
                self.active_items.remove(slice_res.item)

    def check_game_over(self) -> None:
        """Set the game-over flag when conditions are met."""
        if not self.has_initialized or self.is_game_over:
            return
        
        # Only check for time-up (1 minute)
        time_up = time.time() - self.game_start_time > self.game_duration
        if time_up:
            self.is_game_over = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_calibration(self, dt: float, hand_state) -> None:
        cx = self.spawn_config.screen_width // 2
        cy = self.spawn_config.screen_height // 2
        in_zone = abs(hand_state.cx - cx) < 150 and abs(hand_state.cy - cy) < 150

        if in_zone and hand_state.width > 0:
            self.calibration_progress += dt / self.calibration_target_time
            if self.calibration_progress >= 1.0:
                self.has_initialized = True
                self.slice_engine.reset()
                self.game_start_time = time.time()
                self.last_target_rotation_time = time.time()
                self.voice.announce(
                    f"Game Start! Slash the {self.scoreboard.current_target_group.name} foods!"
                )
        else:
            self.calibration_progress = max(0.0, self.calibration_progress - dt * 0.5)

    def _maybe_rotate_target(self) -> None:
        if time.time() - self.last_target_rotation_time <= self.target_rotation_interval:
            return
        food_groups = [g for g in FoodGroup if g != FoodGroup.BOMB]
        current_idx = (
            food_groups.index(self.scoreboard.current_target_group)
            if self.scoreboard.current_target_group in food_groups
            else 0
        )
        next_idx = (current_idx + 1) % len(food_groups)
        self.scoreboard.current_target_group = food_groups[next_idx]
        self.last_target_rotation_time = time.time()
        self.voice.announce(
            f"New target! Find the {self.scoreboard.current_target_group.name} foods!"
        )

    def _maybe_spawn(self) -> None:
        if time.time() - self.last_spawn_time <= self.spawn_interval:
            return
        if not self.catalog:
            return

        num_items = self.spawner.random.randint(1, 4)
        for _ in range(num_items):
            food_def = self.spawner.random.choice(self.catalog)
            self.active_items.append(
                self.spawner.spawn_food(food_def.name, food_def.food_group, food_def.sprite_path)
            )

        if self.trap_catalog and self.spawner.random.random() < self.spawn_config.hazard_chance:
            trap_def = self.spawner.random.choice(self.trap_catalog)
            self.active_items.append(
                self.spawner.spawn_food(trap_def["name"], FoodGroup.BOMB, trap_def["sprite_path"])
            )

        self.last_spawn_time = time.time()

    def _update_physics(self, dt: float) -> None:
        for item in self.active_items[:]:
            item.velocity_y += self.spawn_config.gravity * dt
            item.start_x += item.velocity_x * dt
            item.start_y += item.velocity_y * dt
            item.angle += item.rotation_speed * dt

            if item.start_y > self.spawn_config.screen_height + 100:
                self.active_items.remove(item)
