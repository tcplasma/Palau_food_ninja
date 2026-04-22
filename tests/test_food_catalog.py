import json
from pathlib import Path


def test_food_catalog_contains_all_three_groups() -> None:
    catalog_path = Path("assets/food_groups/catalog.json")
    entries = json.loads(catalog_path.read_text(encoding="utf-8"))
    groups = {entry["food_group"] for entry in entries}
    assert groups == {"protective", "energy", "body_building"}
