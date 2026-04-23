"""Pygame rendering for the Food Ninja game."""

from __future__ import annotations

import cv2
import pygame
from typing import Dict, TYPE_CHECKING

from foodninja.config import SpawnConfig
from foodninja.core.models import TrackMode, TrackingStepResult

if TYPE_CHECKING:
    from foodninja.gameplay.game_state import GameState


class GameRenderer:
    """
    Draws every visual element onto the Pygame screen.

    Receives read-only references each frame — never mutates game state.
    """

    def __init__(self, screen: pygame.Surface, spawn_config: SpawnConfig):
        self.screen = screen
        self.spawn_config = spawn_config
        self.font = pygame.font.SysFont("Arial", 32)
        self.large_font = pygame.font.SysFont("Arial", 64)

    def render_frame(
        self,
        frame,
        tracking_result: TrackingStepResult,
        game_state: GameState,
        food_images: Dict[str, pygame.Surface],
    ) -> None:
        """Compose and blit one complete game frame."""
        sw = self.spawn_config.screen_width
        sh = self.spawn_config.screen_height

        # 1. Camera feed as background
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(rgb_frame.transpose((1, 0, 2)))
        self.screen.blit(frame_surface, (0, 0))

        # 2. Transparent overlay for game elements
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)

        self._draw_trail(overlay, tracking_result)
        self._draw_items(overlay, game_state, food_images)

        if not game_state.has_initialized:
            self._draw_calibration(overlay, game_state)

        self._draw_hud(overlay, game_state)

        if game_state.is_game_over:
            self._draw_game_over(overlay, game_state)

        # 3. Bomb Flash Overlay (Red screen pulse)
        if game_state.bomb_flash_timer > 0:
            flash_surface = pygame.Surface((sw, sh))
            flash_surface.fill((255, 0, 0))
            # Alpha pulse based on timer
            alpha = int(min(1.0, game_state.bomb_flash_timer * 2) * 160)
            flash_surface.set_alpha(alpha)
            overlay.blit(flash_surface, (0, 0))

        self.screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_trail(self, overlay: pygame.Surface, tracking_result: TrackingStepResult) -> None:
        """Cyan meteor trail from recent hand positions."""
        if not hasattr(tracking_result, "trajectory") or not tracking_result.trajectory:
            return
        traj_len = len(tracking_result.trajectory)
        for i, pos in enumerate(tracking_result.trajectory):
            alpha = int((i / traj_len) * 200)
            radius = int(8 + (i / traj_len) * 12)
            color = (0, 255, 255, alpha)
            pygame.draw.circle(overlay, color, (int(pos[0]), int(pos[1])), radius)

    def _draw_items(
        self,
        overlay: pygame.Surface,
        game_state: GameState,
        food_images: Dict[str, pygame.Surface],
    ) -> None:
        """Draw all flying food and bomb sprites."""
        for item in game_state.active_items:
            if item.name in food_images:
                img = food_images[item.name]
                rotated_img = pygame.transform.rotate(img, item.angle)
                rect = rotated_img.get_rect(center=(int(item.start_x), int(item.start_y)))
                overlay.blit(rotated_img, rect.topleft)
            else:
                pygame.draw.circle(
                    overlay, (255, 0, 0), (int(item.start_x), int(item.start_y)), 30
                )

    def _draw_calibration(self, overlay: pygame.Surface, game_state: GameState) -> None:
        """Hand-placement guide and progress bar."""
        cx = self.spawn_config.screen_width // 2
        cy = self.spawn_config.screen_height // 2
        box_color = (0, 255, 0) if game_state.calibration_progress > 0 else (255, 255, 255)

        pygame.draw.rect(overlay, box_color, (cx - 150, cy - 150, 300, 300), 4)
        msg = self.font.render("PLACE HAND HERE TO START", True, box_color)
        overlay.blit(msg, (cx - msg.get_width() // 2, cy - 190))

        bar_width = 300
        pygame.draw.rect(overlay, (50, 50, 50), (cx - 150, cy + 165, bar_width, 15))
        fill = int(bar_width * game_state.calibration_progress)
        pygame.draw.rect(overlay, (0, 255, 0), (cx - 150, cy + 165, fill, 15))

    def _draw_hud(self, overlay: pygame.Surface, game_state: GameState) -> None:
        """Score counters and current target indicator."""
        correct_text = self.font.render(f"Correct: {game_state.scoreboard.correct_hits}", True, (0, 255, 0))
        wrong_text = self.font.render(f"Wrong: {game_state.scoreboard.wrong_hits}", True, (255, 0, 0))
        overlay.blit(correct_text, (20, 20))
        overlay.blit(wrong_text, (200, 20))

        target_name = (
            game_state.scoreboard.current_target_group.name
            if game_state.has_initialized
            else "CALIBRATING..."
        )
        target_text = self.font.render(f"Target: {target_name}", True, (255, 255, 0))
        overlay.blit(target_text, (20, 60))

    def _draw_game_over(self, overlay: pygame.Surface, game_state: GameState) -> None:
        """Darkened overlay with final score summary."""
        sw = self.spawn_config.screen_width
        sh = self.spawn_config.screen_height

        dark = pygame.Surface((sw, sh))
        dark.set_alpha(180)
        dark.fill((0, 0, 0))
        overlay.blit(dark, (0, 0))

        title = self.large_font.render("GAME OVER - SUMMARY", True, (255, 255, 255))
        overlay.blit(title, (sw // 2 - title.get_width() // 2, 200))

        res1 = self.font.render(
            f"Correct Nutrition Matches: {game_state.scoreboard.correct_hits}", True, (0, 255, 0)
        )
        res2 = self.font.render(
            f"Incorrect Matches: {game_state.scoreboard.wrong_hits}", True, (255, 0, 0)
        )
        overlay.blit(res1, (sw // 2 - res1.get_width() // 2, 350))
        overlay.blit(res2, (sw // 2 - res2.get_width() // 2, 400))

        hint = self.font.render("Press Q to quit or close the window.", True, (200, 200, 200))
        overlay.blit(hint, (sw // 2 - hint.get_width() // 2, 550))
