"""
Rectangle class module. Inherits from the Shape class.
"""

from numpy.random import Generator

from simulation.geometry.circle import Circle
from simulation.geometry.constants import TOLERANCE
from simulation.geometry.exceptions import EdgeError
from simulation.geometry.point import Point
from simulation.geometry.shape import Shape

SHAPE_EDGE_COUNT = 4

class Rectangle(Shape):
    """Creates a rectangular shape based on its center, its width and its height."""
    width: float
    height: float
    _perimeter_points: list[Point]
    _edges: list[tuple[Point, Point]]
    _edge_normal_vectors: list[Point]
    _edge_reference_vectors: list[Point]

    def __init__(self, center: Point, width: float, height: float, orientation: float = 0.0, fill: str = "#FFFFFF", outline: str = "#FFFFFF"):
        """Creates a rectangular shape based on its center, its width and its height.
            - center: Point object representing the coordinates of the rectangle's center.
            - width: Floating value representing the size of the rectangle parallel to its local x-axis.
            - height: Floating value representing the size of the rectangle parallel to its local y-axis.
            - orientation (optional): Floating value representing the angle between the rectangle's local x-axis and the simulation's x-axis.
            - fill (optional): String representing the rectangle's background color, in hexadecimal (default white).
            - outline (optional): String representing the rectangle's perimeter color, in hexadecimal (default white)."""
        
        super().__init__(center, orientation, fill, outline)

        if not float(width) > 0.0:
            raise ValueError("Rectangle's width must be bigger then zero.")
        if not float(height) > 0.0:
            raise ValueError("Rectangle's height must be bigger then zero.")
        
        self.width = float(width)
        self.height = float(height)

        self._perimeter_points = [Point(self.width/2.0, self.height/2.0), Point(self.width/2.0, -self.height/2.0),
                                 Point(-self.width/2.0, -self.height/2.0), Point(-self.width/2.0, self.height/2.0)]

        self._edges = [(self._perimeter_points[i-1], self._perimeter_points[i]) for i in range(SHAPE_EDGE_COUNT)]
        self._edge_normal_vectors = [(point2 - point1).rotate(90.0).unit_vector().round(8) for point1, point2 in self._edges]
        self._edge_reference_vectors = [perimeter_point.projection(normal_vector).round(8) for perimeter_point, normal_vector in zip(self._perimeter_points, self._edge_normal_vectors)]

    def contains_point(self, global_point: Point) -> bool:
        local_point = self.translate_to_local(global_point)

        return (-(self.width/2.0 + TOLERANCE) <= local_point.x <= self.width/2.0 + TOLERANCE 
            and -(self.height/2.0 + TOLERANCE) <= local_point.y <= self.height/2.0 + TOLERANCE) 

    def collides_with(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            if self.contains_point(shape.center):
                return True
            else:
                local_circle_center = self.translate_to_local(shape.center)
                closest_point = self.get_closest_point(local_circle_center)

                # Checks if the distance from the closest point to the circle's center is smaller than its radius.
                return (local_circle_center - closest_point).squared_norm() <= (shape.radius)**2.0
        
        elif isinstance(shape, Shape):
            return (any([self.contains_point(corner) for corner in shape.get_perimeter_points()]) 
                 or any([shape.contains_point(corner) for corner in self.get_perimeter_points()]))
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_points(self) -> list[Point]:        
        return [self.translate_to_global(point) for point in self._perimeter_points]
    
    def get_random_point(self, generator: Generator) -> Point:
        if not isinstance(generator, Generator):
            raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
        
        x = generator.uniform(-self.width/2.0, self.width/2.0)
        y = generator.uniform(-self.height/2.0, self.height/2.0)

        return self.translate_to_global(Point(x, y))
    
    def get_closest_point(self, local_point: Point) -> Point:
        # Calculates the closest x and y coordinates on the rectangle's perimeter.
        closest_x = max(-self.width / 2.0, min(local_point.x, self.width / 2.0))
        closest_y = max(-self.height / 2.0, min(local_point.y, self.height / 2.0))

        # Checks if the point is inside the rectangle.
        if self.contains_point(local_point):
            
            # Finds distances to each edge.
            dist_left = abs(local_point.x - (-self.width / 2.0))
            dist_right = abs(local_point.x - (self.width / 2.0))
            dist_top = abs(local_point.y - (self.height / 2.0))
            dist_bottom = abs(local_point.y - (-self.height / 2.0))
            
            # Determines the minimum distances on each axis.
            min_dist_x = min(dist_left, dist_right)
            min_dist_y = min(dist_top, dist_bottom)

            # Prioritizes the vertical edges over the horizontal ones.
            if min_dist_x <= min_dist_y:
                # Prioritizes the left edge over the right one.
                closest_x = -self.width / 2.0 if dist_left < dist_right else self.width / 2.0
            else:
                # Prioritizes the top edge over the bottom one.
                closest_y = -self.height / 2.0 if dist_bottom < dist_top else self.height / 2.0

        return Point(closest_x, closest_y)
    
    def get_edge_normal_vector(self, local_point: Point) -> Point:
        for edge, normal_vector, reference_vector in zip(self._edges, self._edge_normal_vectors, self._edge_reference_vectors):
            if (min(edge[0].x, edge[1].x) - TOLERANCE <= local_point.x <= max(edge[0].x, edge[1].x) + TOLERANCE
            and min(edge[0].y, edge[1].y) - TOLERANCE <= local_point.y <= max(edge[0].y, edge[1].y) + TOLERANCE):
                if local_point.projection(normal_vector) == reference_vector:
                    return normal_vector
        else:
            raise EdgeError("Given point is not on this shape's perimeter. It can't be associated to any edges.")