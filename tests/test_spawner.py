from foodninja.config import SpawnConfig
from foodninja.core.models import FoodGroup
from foodninja.gameplay.spawner import FoodSpawner


def test_spawn_food_creates_upward_motion() -> None:
    spawner = FoodSpawner(SpawnConfig(), seed=7)
    item = spawner.spawn_food(
        name="banana",
        food_group=FoodGroup.PROTECTIVE,
        sprite_path="assets/food_groups/protective/fruits/banana.png",
    )
    assert item.velocity_y < 0
    assert 0 <= item.start_x <= spawner.config.screen_width

