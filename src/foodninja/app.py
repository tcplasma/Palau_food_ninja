import cv2
import json
import pygame
import sys
import time
from typing import List, Optional

from foodninja.config import TrackingConfig, SpawnConfig
from foodninja.core.models import TrackMode, FoodGroup, SpawnItem, TrackingState, TrackingStepResult
from foodninja.core.utils import get_resource_path
from foodninja.tracking.tracker import HandTracker
from foodninja.gameplay.spawner import FoodSpawner
from foodninja.gameplay.slice_engine import SliceEngine, SlicedResult
from foodninja.gameplay.food_catalog import load_food_catalog
from foodninja.gameplay.session import ScoreBoard
from foodninja.gameplay.audio_engine import VoiceAnnouncer, SfxEngine

class FoodNinjaGame:
    def __init__(self):
        pygame.init()
        self.tracking_config = TrackingConfig()
        self.spawn_config = SpawnConfig()
        
        self.screen = pygame.display.set_mode((self.spawn_config.screen_width, self.spawn_config.screen_height))
        pygame.display.set_caption("Food Ninja - Nutrition Education")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32)
        self.large_font = pygame.font.SysFont("Arial", 64)
        
        # Tracking & Game Core
        self.tracker = HandTracker(self.tracking_config)
        self.spawner = FoodSpawner(self.spawn_config)
        self.slice_engine = SliceEngine()
        self.scoreboard = ScoreBoard()
        self.voice = VoiceAnnouncer()
        self.sfx = SfxEngine()
        
        # Load Food Assets
        try:
            self.catalog = load_food_catalog(get_resource_path("assets/food_catalog.json"))
        except Exception as e:
            print(f"Error loading catalog: {e}")
            self.catalog = []
            
        self.food_images = {}
        
        for item in self.catalog:
            try:
                img_path = get_resource_path(item.sprite_path)
                img = pygame.image.load(img_path).convert_alpha()
                scaled_img = pygame.transform.scale(img, (80, 80))
                self.food_images[item.name] = scaled_img
            except Exception as e:
                print(f"Error loading image {item.sprite_path}: {e}")
        
        # Load Trap Catalog (Data-Driven Hazard System)
        self.trap_catalog = []
        try:
            with open(get_resource_path("assets/trap_catalog.json"), "r", encoding="utf-8") as f:
                self.trap_catalog = json.load(f)
            for trap in self.trap_catalog:
                trap_path = get_resource_path(trap["sprite_path"])
                trap_img = pygame.image.load(trap_path).convert_alpha()
                scaled_trap = pygame.transform.scale(trap_img, (80, 80))
                self.food_images[trap["name"]] = scaled_trap
        except Exception as e:
            print(f"Error loading trap catalog: {e}")

        # Game State
        self.active_items: List[SpawnItem] = []
        self.last_spawn_time = time.time()
        self.last_target_rotation_time = time.time()
        self.game_start_time = time.time()
        self.spawn_interval = 2.0 
        self.target_rotation_interval = 15.0 
        self.game_duration = 120.0 # 2 minute round
        self.is_game_over = False
        
        # Calibration State
        self.has_initialized = False
        self.calibration_progress = 0.0 # 0.0 to 1.0
        self.calibration_target_time = 1.5 # seconds to hold hand in zone
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spawn_config.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spawn_config.screen_height)

    def run(self):
        running = True
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # 2. Camera & Tracking Update
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            # Resize if necessary to match screen
            if frame.shape[1] != self.spawn_config.screen_width or frame.shape[0] != self.spawn_config.screen_height:
                frame = cv2.resize(frame, (self.spawn_config.screen_width, self.spawn_config.screen_height))
            
            tracking_result = self.tracker.process_frame(frame)
            
            # 3. Game Logic Update
            dt = self.clock.tick(60) / 1000.0 # Delta time in seconds
            
            if not self.is_game_over:
                self._update_game(dt, tracking_result)

                # Check for slices (Only if game has started)
                if self.has_initialized:
                    slices = self.slice_engine.process_frame(tracking_result.state, self.active_items)
                    for slice_res in slices:
                        if slice_res.item.food_group == self.scoreboard.current_target_group:
                            self.sfx.play_correct()
                        else:
                            self.sfx.play_wrong()
                            
                        self.scoreboard.register_hit(slice_res.item.food_group)
                        if slice_res.item in self.active_items:
                            self.active_items.remove(slice_res.item)
                
                # Check for game over
                if self.has_initialized and (self.scoreboard.wrong_hits >= 10 or (time.time() - self.game_start_time > self.game_duration)):
                    self.is_game_over = True

            # 4. Rendering
            self._render(frame, tracking_result)
            pygame.display.flip()

        self.cap.release()
        pygame.quit()
        sys.exit()

    def _update_game(self, dt: float, tracking_result: TrackingStepResult):
        hand_state = tracking_result.state
        
        # 1. Handle Initialization (Calibration)
        if not self.has_initialized:
            # Check if hand is in the center start zone (300x300)
            cx, cy = self.spawn_config.screen_width // 2, self.spawn_config.screen_height // 2
            in_zone = abs(hand_state.cx - cx) < 150 and abs(hand_state.cy - cy) < 150
            
            # Simplified: any detection in the zone counts
            if in_zone and hand_state.width > 0:
                self.calibration_progress += dt / self.calibration_target_time
                if self.calibration_progress >= 1.0:
                    self.has_initialized = True
                    self.game_start_time = time.time()
                    self.last_target_rotation_time = time.time()
                    self.voice.announce(f"Game Start! Slash the {self.scoreboard.current_target_group.name} foods!")
            else:
                self.calibration_progress = max(0.0, self.calibration_progress - dt * 0.5) # Slow decay
            return

        # 2. Rotate target group (skip non-food groups like BOMB)
        if time.time() - self.last_target_rotation_time > self.target_rotation_interval:
            food_groups = [g for g in FoodGroup if g != FoodGroup.BOMB]
            current_idx = food_groups.index(self.scoreboard.current_target_group) if self.scoreboard.current_target_group in food_groups else 0
            next_idx = (current_idx + 1) % len(food_groups)
            self.scoreboard.current_target_group = food_groups[next_idx]
            self.last_target_rotation_time = time.time()
            # Announcement
            self.voice.announce(f"New target! Find the {self.scoreboard.current_target_group.name} foods!")

        # 3. Spawn food
        if time.time() - self.last_spawn_time > self.spawn_interval:
            if self.catalog:
                # Spawn a wave of 1-4 items
                num_items = self.spawner.random.randint(1, 4)
                for _ in range(num_items):
                    food_def = self.spawner.random.choice(self.catalog)
                    new_item = self.spawner.spawn_food(food_def.name, food_def.food_group, food_def.sprite_path)
                    self.active_items.append(new_item)
                
                # Dedicated Hazard Roll (Data-Driven)
                if self.trap_catalog and self.spawner.random.random() < self.spawn_config.hazard_chance:
                    trap_def = self.spawner.random.choice(self.trap_catalog)
                    trap_item = self.spawner.spawn_food(
                        trap_def["name"], FoodGroup.BOMB, trap_def["sprite_path"]
                    )
                    self.active_items.append(trap_item)
                    
                self.last_spawn_time = time.time()

        # 4. Update item physics
        for item in self.active_items[:]:
            item.velocity_y += self.spawn_config.gravity * dt
            item.start_x += item.velocity_x * dt
            item.start_y += item.velocity_y * dt
            item.angle += item.rotation_speed * dt
            
            if item.start_y > self.spawn_config.screen_height + 100:
                self.active_items.remove(item)

    def _render(self, frame, tracking_result):
        # Convert OpenCV frame to Pygame surface
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Transpose since pygame expects (width, height)
        frame_surface = pygame.surfarray.make_surface(rgb_frame.transpose((1, 0, 2)))
        self.screen.blit(frame_surface, (0, 0))
        
        # Transparent overlay for game elements
        overlay = pygame.Surface((self.spawn_config.screen_width, self.spawn_config.screen_height), pygame.SRCALPHA)
        
        # Draw Meteor Trail
        if hasattr(tracking_result, 'trajectory') and tracking_result.trajectory:
            traj_len = len(tracking_result.trajectory)
            for i, pos in enumerate(tracking_result.trajectory):
                # Fade out: older points are smaller and more transparent
                alpha = int((i / traj_len) * 200)
                radius = int(8 + (i / traj_len) * 12)
                color = (0, 255, 255, alpha) # Cyan trail
                pygame.draw.circle(overlay, color, (int(pos[0]), int(pos[1])), radius)

        # Draw Food & Bombs
        for item in self.active_items:
            if item.name in self.food_images:
                img = self.food_images[item.name]
                # Real-time smooth rotation
                rotated_img = pygame.transform.rotate(img, item.angle)
                rect = rotated_img.get_rect(center=(int(item.start_x), int(item.start_y)))
                overlay.blit(rotated_img, rect.topleft)
            else:
                pygame.draw.circle(overlay, (255, 0, 0), (int(item.start_x), int(item.start_y)), 30)

        # Draw Hand Tracker (Always visible for feedback)
        tracking_color = (0, 255, 0) # Default Green
        if not self.has_initialized or tracking_result.mode == TrackMode.REINIT:
            tracking_color = (255, 255, 0) # Yellow during searching/calibration
        elif tracking_result.mode == TrackMode.LOST:
            tracking_color = (255, 0, 0)   # Red during 1s delay loss

        if tracking_result.state and tracking_result.state.width > 0:
            # We skip drawing the developer bounding box and center dot for a cleaner look
            pass

        # Draw Calibration Progress
        if not self.has_initialized:
            cx, cy = self.spawn_config.screen_width // 2, self.spawn_config.screen_height // 2
            box_color = (0, 255, 0) if self.calibration_progress > 0 else (255, 255, 255)
            pygame.draw.rect(overlay, box_color, (cx - 150, cy - 150, 300, 300), 4)
            msg = self.font.render("PLACE HAND HERE TO START", True, box_color)
            overlay.blit(msg, (cx - msg.get_width() // 2, cy - 190))
            
            bar_width = 300
            pygame.draw.rect(overlay, (50, 50, 50), (cx - 150, cy + 165, bar_width, 15))
            pygame.draw.rect(overlay, (0, 255, 0), (cx - 150, cy + 165, int(bar_width * self.calibration_progress), 15))

        # Debug HUD (Hidden for production)
        # fps = int(self.clock.get_fps())
        # debug_msg = f"FPS: {fps} | MODE: {tracking_result.mode.value.upper()} | CONF: {tracking_result.confidence:.2f}"
        # debug_surface = self.font.render(debug_msg, True, (255, 255, 255))
        # overlay.blit(debug_surface, (self.spawn_config.screen_width - debug_surface.get_width() - 20, self.spawn_config.screen_height - 40))
        
        # UI Text
        correct_text = self.font.render(f"Correct: {self.scoreboard.correct_hits}", True, (0, 255, 0))
        wrong_text = self.font.render(f"Wrong: {self.scoreboard.wrong_hits}", True, (255, 0, 0))
        overlay.blit(correct_text, (20, 20))
        overlay.blit(wrong_text, (200, 20))
        
        target_name = self.scoreboard.current_target_group.name if self.has_initialized else "CALIBRATING..."
        target_text = self.font.render(f"Target: {target_name}", True, (255, 255, 0))
        overlay.blit(target_text, (20, 60))

        # Status Messages (Hidden for cleaner experience)
        pass

        if self.is_game_over:
            # Draw blur or darkened overlay
            s = pygame.Surface((self.spawn_config.screen_width, self.spawn_config.screen_height))
            s.set_alpha(180)
            s.fill((0, 0, 0))
            overlay.blit(s, (0, 0))
            
            title = self.large_font.render("GAME OVER - SUMMARY", True, (255, 255, 255))
            overlay.blit(title, (self.spawn_config.screen_width // 2 - title.get_width() // 2, 200))
            
            res1 = self.font.render(f"Correct Nutrition Matches: {self.scoreboard.correct_hits}", True, (0, 255, 0))
            res2 = self.font.render(f"Incorrect Matches: {self.scoreboard.wrong_hits}", True, (255, 0, 0))
            
            overlay.blit(res1, (self.spawn_config.screen_width // 2 - res1.get_width() // 2, 350))
            overlay.blit(res2, (self.spawn_config.screen_width // 2 - res2.get_width() // 2, 400))
            
            hint = self.font.render("Press Q to quit or close the window.", True, (200, 200, 200))
            overlay.blit(hint, (self.spawn_config.screen_width // 2 - hint.get_width() // 2, 550))

        self.screen.blit(overlay, (0, 0))

if __name__ == "__main__":
    game = FoodNinjaGame()
    game.run()
