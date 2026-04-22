import random

from foodninja.config import SpawnConfig
from foodninja.core.models import FoodGroup, SpawnItem


class FoodSpawner:
    def __init__(self, config: SpawnConfig, seed: int | None = None) -> None:
        self.config = config
        self.random = random.Random(seed)

    def spawn_food(self, name: str, food_group: FoodGroup, sprite_path: str) -> SpawnItem:
        return SpawnItem(
            name=name,
            food_group=food_group,
            sprite_path=sprite_path,
            start_x=self.random.uniform(0.15, 0.85) * self.config.screen_width,
            start_y=self.config.screen_height + 20.0,
            velocity_x=self.random.uniform(
                self.config.min_launch_speed_x,
                self.config.max_launch_speed_x,
            ),
            velocity_y=-self.random.uniform(
                self.config.min_launch_speed_y,
                self.config.max_launch_speed_y,
            ),
            angle=self.random.uniform(0, 360),
            rotation_speed=self.random.uniform(-180, 180) # Degrees per second
        )

