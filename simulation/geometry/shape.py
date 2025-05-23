"""
Base class module. 

The Shape class defined here should not directly be used and will make the simulation fail.  
"""

from __future__ import annotations

from numpy.random import Generator
import re

from simulation.geometry.vector import Vector2D

HEX_PATTERN = r"^#([A-Fa-f0-9]{6})$"

class Shape:
    """Base class for all Shape objects."""
    center: Vector2D
    orientation: float
    fill: str
    outline: str

    def __init__(self, center: Vector2D, orientation: float, fill: str, outline: str):
        """Base class for all Shape objects.
            - center: Point object representing the center of the shape object.
            - orientation: Floating value representing the angle between the shape's local x-axis and the simulation's x-axis.
            - fill: String representing the shape's background color, in hexadecimal.
            - outline: String representing the shape's perimeter color, in hexadecimal."""
        
        if not isinstance(center, Vector2D):
            raise TypeError(f"unsupported parameter type(s) for center: '{type(center).__name__}'")
        if not bool(re.match(HEX_PATTERN, str(fill))):
            raise ValueError(f"Unsupported hexadecimal pattern for fill ({fill}).")
        if not bool(re.match(HEX_PATTERN, str(outline))):
            raise ValueError(f"Unsupported hexadecimal pattern for outline ({outline}).")
        
        self.center = center
        self.orientation = float(orientation)
        self.fill = str(fill)
        self.outline = str(outline)

    def __eq__(self, other) -> bool:
        """Checks if two Shape objects are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False
        
    def __repr__(self) -> str:
        """Shape object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Shape object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def copy(self) -> Shape:
        """Returns a copy of the object as a new instance of the same class."""
        return eval(repr(self))

    def move_center(self, translation: Vector2D) -> Shape:
        """Moves the center of this shape by a given translation vector, represented by a Point object."""
        self.center += translation
        return self

    def rotate(self, angle: float) -> Shape:
        """Rotates this shape around its center by the given angle (in degrees)."""
        self.orientation += float(angle)
        return self

    def translate_to_local(self, global_point: Vector2D) -> Vector2D:
        """Translates a point from the simulation's global coordinates to the shape's local coordinates."""
        return (global_point - self.center).rotate(-self.orientation)

    def translate_to_global(self, local_point: Vector2D) -> Vector2D:
        """Translates a point from the shape's local coordinates to the simulation's global coordinates."""
        return local_point.rotate(self.orientation) + self.center
    
    def contains_point(self, global_point: Vector2D) -> bool:
        """Checks if a global Point object lies inside this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def collides_with(self, shape: Shape) -> bool:
        """Checks if another Shape object collides with this shape."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_perimeter_points(self) -> list[Vector2D]:
        """Returns a list of points that forms the corners of this shape's perimeter."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_random_point(self, generator: Generator) -> Vector2D:
        """Return a random Point object contained within this shape, using the generator object to generate random values."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_closest_point(self, local_point: Vector2D) -> Vector2D:
        """Returns the closest point on the shape's perimeter from another local point."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_edge_normal_vector(self, local_point: Vector2D) -> Vector2D:
        """Returns the normal vector of the edge that the given local point is on."""
        raise NotImplementedError("Subclasses must implement this method")
    
