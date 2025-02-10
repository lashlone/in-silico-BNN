"""
Base class module. 

The Shape class defined here should not directly be used as the shape parameter of Element objects and will make the simulation fail.  
"""

from typing import List

from simulation.geometry.point import Point

class Shape():
    """Base class for all Shape objects."""

    def __init__(self, center: Point, orientation: float):
        """Base class for all Shape objects."""
        if not isinstance(center, Point):
            raise TypeError(f"unsupported parameter type(s) for center: '{type(center).__name__}'")
        
        self.center = center
        self.orientation = float(orientation)

    def move_center(self, translation: Point):
        """Moves the center of this shape by a given translation vector, represented by a Point object."""
        self.center += translation

    def rotate(self, angle: float):
        """Rotates this shape around its center by the given angle (in degrees)."""
        self.orientation += float(angle)

    def translate_to_local(self, point: Point) -> Point:
        """Translates a point from the simulation's global coordinates to the shape's local coordinates."""
        return (point - self.center).rotate(-self.orientation)

    def translate_to_global(self, point: Point) -> Point:
        """Translates a point from the shape's local coordinates to the simulation's global coordinates."""
        return point.rotate(self.orientation) + self.center
    
    def contains_point(self, point: Point) -> bool:
        """Checks if a Point object lies inside this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def collides_width(self, shape) -> bool:
        """Checks if another Shape object collides with this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_perimeter_corners(self) -> List[Point]:
        """Returns a list of points that forms the corners of this shape's perimeter."""
        raise NotImplementedError("Subclasses must implement this method.")
