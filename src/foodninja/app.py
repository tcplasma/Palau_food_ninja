"""
Food Ninja — Main application entry point.

Thin coordination layer that wires together:
- Camera capture (OpenCV)
- Hand tracking (HandTracker)
- Game logic (GameState)
- Rendering (GameRenderer)
"""

import cv2
import json
import pygame
import sys

from foodninja.config import TrackingConfig, SpawnConfig
from foodninja.core.models import FoodGroup
from foodninja.core.utils import get_resource_path
from foodninja.tracking.tracker import HandTracker
from foodninja.gameplay.food_catalog import load_food_catalog
from foodninja.gameplay.game_state import GameState
from foodninja.gameplay.renderer import GameRenderer


class FoodNinjaGame:
    def __init__(self):
        pygame.init()
        self.tracking_config = TrackingConfig()
        self.spawn_config = SpawnConfig()

        self.screen = pygame.display.set_mode(
            (self.spawn_config.screen_width, self.spawn_config.screen_height)
        )
        pygame.display.set_caption("Food Ninja - Nutrition Education")
        self.clock = pygame.time.Clock()

        # Tracking
        self.tracker = HandTracker(self.tracking_config)

        # Load catalogs
        try:
            catalog = load_food_catalog(get_resource_path("assets/food_catalog.json"))
        except Exception as e:
            print(f"Error loading catalog: {e}")
            catalog = []

        trap_catalog: list[dict] = []
        try:
            with open(get_resource_path("assets/trap_catalog.json"), "r", encoding="utf-8") as f:
                trap_catalog = json.load(f)
        except Exception as e:
            print(f"Error loading trap catalog: {e}")

        # Game state (owns spawner, slice engine, scoreboard, audio)
        self.game_state = GameState(self.spawn_config, catalog, trap_catalog)

        # Renderer
        self.renderer = GameRenderer(self.screen, self.spawn_config)

        # Load sprite images for rendering
        self.food_images: dict[str, pygame.Surface] = {}
        for item in catalog:
            try:
                img = pygame.image.load(get_resource_path(item.sprite_path)).convert_alpha()
                self.food_images[item.name] = pygame.transform.scale(img, (80, 80))
            except Exception as e:
                print(f"Error loading image {item.sprite_path}: {e}")

        for trap in trap_catalog:
            try:
                img = pygame.image.load(get_resource_path(trap["sprite_path"])).convert_alpha()
                self.food_images[trap["name"]] = pygame.transform.scale(img, (80, 80))
            except Exception as e:
                print(f"Error loading trap image {trap['sprite_path']}: {e}")

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spawn_config.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spawn_config.screen_height)

    def run(self):
        running = True
        while running:
            # 1. Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            # 2. Camera & tracking
            dt = self.clock.tick(60) / 1000.0
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            if (frame.shape[1] != self.spawn_config.screen_width
                    or frame.shape[0] != self.spawn_config.screen_height):
                frame = cv2.resize(frame, (self.spawn_config.screen_width, self.spawn_config.screen_height))

            tracking_result = self.tracker.process_frame(frame, dt=dt)

            # 3. Game logic
            self.game_state.update(dt, tracking_result)
            self.game_state.process_slices(tracking_result)
            self.game_state.check_game_over()

            # 4. Render
            self.renderer.render_frame(frame, tracking_result, self.game_state, self.food_images)
            pygame.display.flip()

        self.cap.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = FoodNinjaGame()
    game.run()
