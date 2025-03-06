"""
Base class module. 

The Shape class defined here should not directly be used as the shape parameter of Element objects and will make the simulation fail.  
"""

from __future__ import annotations

from numpy.random import Generator
import re

from simulation.geometry.point import Point

HEX_PATTERN = r"^#([A-Fa-f0-9]{6})$"

class Shape:
    """Base class for all Shape objects."""
    center: Point
    orientation: float
    fill: str
    stroke: str

    def __init__(self, center: Point, orientation: float, fill: str, stroke: str):
        """Base class for all Shape objects."""
        if not isinstance(center, Point):
            raise TypeError(f"unsupported parameter type(s) for center: '{type(center).__name__}'")
        if not bool(re.match(HEX_PATTERN, str(fill))):
            raise ValueError(f"Unsupported hexadecimal pattern for fill ({fill}).")
        if not bool(re.match(HEX_PATTERN, str(stroke))):
            raise ValueError(f"Unsupported hexadecimal pattern for stroke ({stroke}).")
        
        self.center = center
        self.orientation = float(orientation)
        self.fill = str(fill)
        self.stroke = str(stroke)

    def __eq__(self, other) -> bool:
        """Checks if two Shape are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False
        
    def __repr__(self) -> str:
        """Object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"

    def move_center(self, translation: Point) -> None:
        """Moves the center of this shape by a given translation vector, represented by a Point object."""
        self.center += translation

    def rotate(self, angle: float) -> None:
        """Rotates this shape around its center by the given angle (in degrees)."""
        self.orientation += float(angle)

    def translate_to_local(self, global_point: Point) -> Point:
        """Translates a point from the simulation's global coordinates to the shape's local coordinates."""
        return (global_point - self.center).rotate(-self.orientation)

    def translate_to_global(self, local_point: Point) -> Point:
        """Translates a point from the shape's local coordinates to the simulation's global coordinates."""
        return local_point.rotate(self.orientation) + self.center
    
    def contains_point(self, global_point: Point) -> bool:
        """Checks if a global Point object lies inside this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def collides_with(self, shape: Shape) -> bool:
        """Checks if another Shape object collides with this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_perimeter_corners(self) -> list[Point]:
        """Returns a list of points that forms the corners of this shape's perimeter."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_random_point(self, genenrator: Generator) -> Point:
        """Return a random Point object contained within this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_closest_point(self, local_point: Point) -> Point:
        """Returns the closest point on the shape's perimeter from another local point."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_edge_normal_vector(self, local_point: Point) -> Point:
        """Returns the normal vector of the edge that the given local point is on."""
        raise NotImplementedError("Subclasses must implement this method")
    
