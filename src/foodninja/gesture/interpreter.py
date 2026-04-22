from dataclasses import dataclass


@dataclass(slots=True)
class GestureDecision:
    is_swipe: bool
    is_hit: bool
    speed: float


def interpret_linear_motion(speed: float, swipe_threshold: float, hit_threshold: float) -> GestureDecision:
    return GestureDecision(
        is_swipe=speed >= swipe_threshold,
        is_hit=speed >= hit_threshold,
        speed=speed,
    )

