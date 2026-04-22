import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from foodninja.core.models import SpawnItem, TrackingState

@dataclass(slots=True)
class SlicedResult:
    item: SpawnItem
    slice_point: Tuple[float, float]
    slice_angle: float

class SliceEngine:
    """
    Handles collision detection between the tracked hand trajectory 
    and the flying food items.
    """
    def __init__(self, slice_threshold: float = 30.0):
        self.slice_threshold = slice_threshold
        self.previous_pos: Optional[Tuple[float, float]] = None

    def process_frame(self, hand_state: TrackingState, active_items: List[SpawnItem]) -> List[SlicedResult]:
        """
        Checks if the movement from previous_pos to current hand_state 
        intersects with any active items.
        """
        current_pos = (hand_state.cx, hand_state.cy)
        sliced_items: List[SlicedResult] = []
        
        # Use the tracker's dynamic bounding box size to determine slice thickness
        dynamic_radius = max(30.0, hand_state.width / 2.0)

        if self.previous_pos is not None:
            for item in active_items:
                if self._check_intersection(self.previous_pos, current_pos, item, dynamic_radius):
                    # Calculate angle of the slice trajectory
                    angle = math.atan2(current_pos[1] - self.previous_pos[1], 
                                       current_pos[0] - self.previous_pos[0])
                    
                    sliced_items.append(SlicedResult(
                        item=item,
                        slice_point=current_pos,
                        slice_angle=math.degrees(angle)
                    ))

        self.previous_pos = current_pos
        return sliced_items

    def _check_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], item: SpawnItem, hit_radius: float) -> bool:
        """
        Checks if the fast-moving hand trajectory segment p1-p2 intersects the item.
        Uses point-to-line-segment distance for robust high-speed collision detection.
        """
        ix, iy = item.start_x, item.start_y
        
        # Vector math for point-to-segment distance
        l2 = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        
        if l2 == 0:
            # p1 == p2, just check point distance
            dist2 = (ix - p1[0])**2 + (iy - p1[1])**2
            return dist2 <= hit_radius**2
            
        # Consider the line extending the segment, parameterized as p1 + t (p2 - p1).
        # We find projection of point p onto the line. 
        # It falls where t = [(p-p1) . (p2-p1)] / |p2-p1|^2
        t = max(0, min(1, ((ix - p1[0]) * (p2[0] - p1[0]) + (iy - p1[1]) * (p2[1] - p1[1])) / l2))
        
        projection = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        
        dist2 = (ix - projection[0])**2 + (iy - projection[1])**2
        return dist2 <= hit_radius**2

