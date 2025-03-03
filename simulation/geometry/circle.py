"""
Circle class module. Inherits from the Shape class.
"""

import numpy as np
from math import pi, cos, sin

from simulation.geometry.constants import TOLERANCE
from simulation.geometry.exceptions import CurvedEdgeError, EdgeError
from simulation.geometry.point import Point
from simulation.geometry.shape import Shape

class Circle(Shape):
    """Creates a circular shape based on its center and its radius."""
    radius: float
    
    def __init__(self, center: Point, radius: float, generator: None | np.random.Generator = None):
        """Creates a circular shape based on its center and its radius.
            - center: the center's coordinates of the circle.
            - radius: radius of the circle object.
            - generator (optional): Generator object to use when generating random values.
        """
        super().__init__(center, 0.0, generator)
        self.radius = float(radius)

    def contains_point(self, global_point: Point) -> bool:
        return (global_point - self.center).squared_norm() <= (self.radius + TOLERANCE) **2.0

    def collides_with(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            return (shape.center - self.center).squared_norm() <= (self.radius + shape.radius + 2.0*TOLERANCE)**2.0
        
        elif isinstance(shape, Shape):
            return shape.collides_with(self)
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_corners(self):
        raise CurvedEdgeError("The corners of curved polygons are not defined")
    
    def get_random_point(self) -> Point:
        radius = self.generator.uniform(0.0, self.radius)
        orientation = self.generator.uniform(0.0, 2.0*pi)

        return self.translate_to_global(Point(radius*cos(orientation), radius*sin(orientation)))

    def get_closest_point(self, local_point: Point) -> Point:
        return (self.radius/local_point.norm()) * local_point
    
    def get_edge_normal_vector(self, local_point: Point) -> Point:
        if local_point.squared_norm() - self.radius ** 2.0 <= TOLERANCE ** 2.0:
            return local_point.unit_vector()
        else:
            raise EdgeError("Given point is not on this shape's perimeter. It won't be associated to any normal vector.")

        