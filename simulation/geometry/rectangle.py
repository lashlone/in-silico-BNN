"""
Rectangle class module. Inherits from the Shape class.
"""

import numpy as np

from simulation.geometry.exceptions import EdgeError
from simulation.geometry.shape import Shape
from simulation.geometry.point import Point
from simulation.geometry.circle import Circle

SHAPE_EDGE_COUNT = 4

class Rectangle(Shape):
    """Creates a rectangular shape based on its center, its width and its height."""
    width: float
    height: float
    perimeter_points: list[Point]
    edges: list[tuple[Point, Point]]
    edge_normal_vectors: list[Point]
    edge_reference_vectors: list[Point]

    def __init__(self, center: Point, width: float, height: float, orientation: float = 0.0, generator: None | np.random.Generator = None):
        """
        Creates a rectangular shape based on its center, its width and its height.
            - center: the center's coordinates of the rectangle.
            - width: size of the rectangle on the x-axis.
            - height: size of the rectangle on the y-axis.
            - orientation (optional): angle between the shape's local x-axis and the simulation's x-axis.
            - generator (optional): Generator object to use when generating random values.
        """
        super().__init__(center, orientation, generator)
        self.width = float(width)
        self.height = float(height)

        self.perimeter_points = [Point(self.width/2.0, self.height/2.0), Point(self.width/2.0, -self.height/2.0),
                                 Point(-self.width/2.0, -self.height/2.0), Point(-self.width/2.0, self.height/2.0)]

        self.edges = [(self.perimeter_points[i-1], self.perimeter_points[i]) for i in range(SHAPE_EDGE_COUNT)]
        self.edge_normal_vectors = [(point2 - point1).rotate(90.0).unit_vector() for point1, point2 in self.edges]
        self.edge_reference_vectors = [perimeter_point.projection(normal_vector) for perimeter_point, normal_vector in zip(self.perimeter_points, self.edge_normal_vectors)]

    def contains_point(self, point: Point) -> bool:
        local_point = self.translate_to_local(point)

        return (-self.width/2.0 <= local_point.x <= self.width/2.0 
            and -self.height/2.0 <= local_point.y <= self.height/2.0) 

    def collides_with(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            local_circle_center = self.translate_to_local(shape.center)
            closest_point = self.get_closest_point(local_circle_center)

            # Checks if the distance from the closest point to the circle's center is smaller than its radius.
            return (local_circle_center - closest_point).squared_norm() <= (shape.radius)**2.0
        
        elif isinstance(shape, Shape):
            return (any([self.contains_point(corner) for corner in shape.get_perimeter_corners()]) 
                 or any([shape.contains_point(corner) for corner in self.get_perimeter_corners()]))
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_corners(self) -> list[Point]:        
        return [self.translate_to_global(point) for point in self.perimeter_points]
    
    def get_random_point(self) -> Point:
        x = self.generator.uniform(-self.width/2.0, self.width/2.0)
        y = self.generator.uniform(-self.height/2.0, self.height/2.0)

        return self.translate_to_global(Point(x, y))
    
    def get_closest_point(self, local_point):
        closest_x = max(-self.width/2.0, min(local_point.x, self.width/2.0))
        closest_y = max(-self.height/2.0, min(local_point.y, self.height/2.0))
        return Point(closest_x, closest_y)
    
    def get_edge_normal_vector(self, local_point):
        for normal_vector, reference_vector in zip(self.edge_normal_vectors, self.edge_reference_vectors):
            if local_point.projection(normal_vector) == reference_vector:
                return normal_vector
        else:
            raise EdgeError("Given point is not on this shape's perimeter. It can't be associated to any edges.")