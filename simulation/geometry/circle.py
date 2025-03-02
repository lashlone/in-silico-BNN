"""
Circle class module. Inherits from the Shape class.
"""

from math import pi, cos, sin

from simulation.geometry.shape import Shape
from simulation.geometry.point import Point
from simulation.geometry.exceptions import CurvedLineError

class Circle(Shape):
    """Creates a circular shape based on its center and its radius."""
    radius: float
    
    def __init__(self, center: Point, radius: float):
        """Creates a circular shape based on its center and its radius."""
        super().__init__(center, 0.0)
        self.radius = float(radius)

    def contains_point(self, point: Point) -> bool:
        return (point - self.center).squared_norm() <= self.radius**2.0

    def collides_with(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            return (shape.center - self.center).squared_norm() <= (self.radius + shape.radius)**2.0
        
        elif isinstance(shape, Shape):
            return shape.collides_with(self)
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_corners(self):
        raise CurvedLineError("The corners of curved polygons are not defined")
    
    def get_random_point(self) -> Point:
        radius = self.generator.uniform(0.0, self.radius)
        orientation = self.generator.uniform(0.0, 2.0*pi)

        return self.translate_to_global(Point(radius*cos(orientation), radius*sin(orientation)))


        