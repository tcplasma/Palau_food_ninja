from dataclasses import dataclass, field

from foodninja.core.models import FoodGroup


@dataclass(slots=True)
class ScoreBoard:
    correct_hits: int = 0
    wrong_hits: int = 0
    current_target_group: FoodGroup = FoodGroup.PROTECTIVE
    history: list[str] = field(default_factory=list)

    def register_hit(self, group: FoodGroup) -> None:
        """Register a food slice. Bombs must use register_bomb_hit() instead."""
        if group == self.current_target_group:
            self.correct_hits += 1
            self.history.append(f"correct:{group.value}")
        else:
            self.wrong_hits += 1
            self.history.append(f"wrong:{group.value}")

    def register_bomb_hit(self, penalty: int = 3) -> None:
        """Register a bomb hit with the given penalty score."""
        self.wrong_hits += penalty
        self.history.append("bomb_hit")

