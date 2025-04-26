"""
Circle class module. Inherits from the Shape class.
"""

from math import pi, cos, sin
from numpy.random import Generator

from simulation.geometry.constants import TOLERANCE
from simulation.geometry.exceptions import CurvedEdgeError, EdgeError
from simulation.geometry.vector import Vector2D
from simulation.geometry.shape import Shape

class Circle(Shape):
    """Defines a circular shape based on its center and its radius."""
    radius: float
    
    def __init__(self, center: Vector2D, radius: float, orientation: float = 0.0, fill: str = "#FFFFFF", outline: str = "#FFFFFF"):
        """Defines a circular shape based on its center and its radius.
            - center: Point object representing the coordinates of the circle's center.
            - radius: Floating value representing the radius of the circle object.
            - orientation (optional): Floating value representing the orientation of the circle object. This parameter is inherited from the shape class but is not used.
            - fill (optional): String representing the circle's background color, in hexadecimal (default white).
            - outline (optional): String representing the circle's perimeter color, in hexadecimal (default white)."""
        
        super().__init__(center, orientation, fill, outline)

        if not float(radius) > 0.0:
            raise ValueError("Circle's radius must be bigger then zero.")
        
        self.radius = float(radius)
        
    def contains_point(self, global_point: Vector2D) -> bool:
        return (global_point - self.center).squared_norm() <= (self.radius + TOLERANCE) **2.0

    def collides_with(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            return (shape.center - self.center).squared_norm() <= (self.radius + shape.radius + 2.0*TOLERANCE)**2.0
        
        elif isinstance(shape, Shape):
            return shape.collides_with(self)
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_points(self):
        raise CurvedEdgeError("The corners of curved polygons are not defined")
    
    def get_random_point(self, generator: Generator) -> Vector2D:
        if not isinstance(generator, Generator):
            raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
        
        radius = generator.uniform(0.0, self.radius)
        orientation = generator.uniform(0.0, 2.0*pi)

        return self.translate_to_global(Vector2D(radius*cos(orientation), radius*sin(orientation)))

    def get_closest_point(self, local_point: Vector2D) -> Vector2D:
        return (self.radius/local_point.norm()) * local_point
    
    def get_edge_normal_vector(self, local_point: Vector2D) -> Vector2D:
        if local_point.squared_norm() - self.radius ** 2.0 <= TOLERANCE ** 2.0:
            return local_point.unit_vector()
        else:
            raise EdgeError("Given point is not on this shape's perimeter. It won't be associated to any normal vector.")

        