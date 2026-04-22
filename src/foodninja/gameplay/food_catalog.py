import json
from dataclasses import dataclass
from pathlib import Path

from foodninja.core.models import FoodGroup


@dataclass(slots=True)
class FoodDefinition:
    name: str
    food_group: FoodGroup
    sprite_path: str


def load_food_catalog(catalog_path: str | Path) -> list[FoodDefinition]:
    raw_entries = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    return [
        FoodDefinition(
            name=entry["name"],
            food_group=FoodGroup(entry["food_group"]),
            sprite_path=entry["sprite_path"],
        )
        for entry in raw_entries
    ]

