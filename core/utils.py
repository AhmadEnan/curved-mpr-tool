import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class ControlPointManager:
    """Manages control points for curved MPR."""
    
    def __init__(self):
        self.points_world: List[np.ndarray] = []  # Each point is (3,) array in world coords
        self.selected_idx: int = -1
    
    def add_point(self, point_world: np.ndarray):
        """Add a control point (world coordinates)."""
        self.points_world.append(np.array(point_world, dtype=np.float64))
        logger.debug(f"Added point {len(self.points_world)}: {point_world}")
    
    def remove_point(self, idx: int):
        """Remove point at index."""
        if 0 <= idx < len(self.points_world):
            self.points_world.pop(idx)
            if self.selected_idx == idx:
                self.selected_idx = -1
            elif self.selected_idx > idx:
                self.selected_idx -= 1
            logger.debug(f"Removed point at index {idx}")
    
    def move_point(self, idx: int, new_position_world: np.ndarray):
        """Update point position."""
        if 0 <= idx < len(self.points_world):
            self.points_world[idx] = np.array(new_position_world, dtype=np.float64)
            logger.debug(f"Moved point {idx} to {new_position_world}")
    
    def get_points_array(self) -> np.ndarray:
        """Get all points as (N, 3) array."""
        if len(self.points_world) == 0:
            return np.empty((0, 3))
        return np.array(self.points_world)
    
    def clear(self):
        """Remove all points."""
        self.points_world.clear()
        self.selected_idx = -1
        logger.debug("Cleared all points")
    
    def select_point(self, idx: int):
        """Select point at index."""
        if 0 <= idx < len(self.points_world):
            self.selected_idx = idx
        else:
            self.selected_idx = -1
    
    def get_selected(self) -> int:
        """Get selected point index (-1 if none)."""
        return self.selected_idx