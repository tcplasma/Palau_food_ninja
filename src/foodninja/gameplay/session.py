from dataclasses import dataclass, field

from foodninja.core.models import FoodGroup


@dataclass(slots=True)
class ScoreBoard:
    correct_hits: int = 0
    wrong_hits: int = 0
    current_target_group: FoodGroup = FoodGroup.PROTECTIVE
    history: list[str] = field(default_factory=list)

    def register_hit(self, group: FoodGroup) -> None:
        if group == FoodGroup.BOMB:
            # Slicing a bomb is a major penalty
            self.wrong_hits += 3
            self.history.append("bomb_hit")
        elif group == self.current_target_group:
            self.correct_hits += 1
            self.history.append(f"correct:{group.value}")
        else:
            self.wrong_hits += 1
            self.history.append(f"wrong:{group.value}")

