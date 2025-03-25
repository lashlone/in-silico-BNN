"""
Triangle class module. Inherits from the Shape class.
"""

from math import atan, degrees, tan
from numpy.random import Generator

from simulation.geometry.circle import Circle
from simulation.geometry.constants import TOLERANCE
from simulation.geometry.exceptions import EdgeError
from simulation.geometry.shape import Shape
from simulation.geometry.point import Point

SHAPE_EDGE_COUNT = 3

class IsoscelesTriangle(Shape):
    """Creates a isosceles triangular shape based on its center, its base and its height."""
    base: float
    height: float
    _perimeter_points: list[Point]
    _edges: list[tuple[Point, Point]]
    _edge_normal_vectors: list[Point]
    _edge_reference_vectors: list[Point]

    def __init__(self, center: Point, base: float, height: float, orientation: float = 0.0, fill: str = "#FFFFFF", outline: str = "#FFFFFF"):
        """Creates isosceles triangular shape based on its center, its base and its height.
            - center: Point object representing the center's coordinates of the rectangle boxing the triangle.
            - base: Floating value representing the size of the isosceles triangle's base, parallel to its local y-axis.
            - height: Floating value representing the size of the isosceles triangle's height, parallel to its local x-axis.
            - orientation (optional): Floating value representing the angle between the triangle's local x-axis and the simulation's x-axis.
            - fill (optional): String representing the triangle's background color, in hexadecimal (default white).
            - outline (optional): String representing the triangle's perimeter color, in hexadecimal (default white)."""
        
        super().__init__(center, orientation, fill, outline)
        
        if not float(base) > 0.0:
            raise ValueError("Triangle's base must be bigger then zero.")
        if not float(height) > 0.0:
            raise ValueError("Triangle's height must be bigger then zero.")
        
        self.base = float(base)
        self.height = float(height)
        
        self._perimeter_points = [Point(self.height/2.0, 0.0),
                                  Point(-self.height/2.0, -self.base/2.0),
                                  Point(-self.height/2.0, self.base/2.0)]
        
        self._edges = [(self._perimeter_points[i-1], self._perimeter_points[i]) for i in range(SHAPE_EDGE_COUNT)]
        self._edge_normal_vectors = [(point2 - point1).rotate(90.0).unit_vector() for point1, point2 in self._edges]
        self._edge_reference_vectors = [perimeter_point.projection(normal_vector) for perimeter_point, normal_vector in zip(self._perimeter_points, self._edge_normal_vectors)]

    def get_barycentric_coordinates(self, local_point: Point) -> list[float]:
        """Returns the barycentric coordinates of the given point, using the triangle's three vertices as base."""
        v0, v1, v2 = self._perimeter_points

        # Defines a linear system with the condition on the sum of the lambdas. See references for more explications.
        a1 = v1 - v0
        a2 = v2 - v0
        b = local_point - v0

        # Uses Cramer's rule in 2D to quickly solve the linear system.
        det_a = a1.x*a2.y - a1.y*a2.x

        lambda1 = (b.x*a2.y - b.y*a2.x)/(det_a)
        lambda2 = (a1.x*b.y - a1.y*b.x)/(det_a)

        # Solves for lambda zero with the condition on the sum of the lambdas.
        lambda0 = 1.0 - lambda1 - lambda2

        return lambda0, lambda1, lambda2
    
    def contains_point(self, global_point: Point) -> bool:
        local_point = self.translate_to_local(global_point)
        local_barycentric_point = self.get_barycentric_coordinates(local_point)

        return all([0.0 <= value <= 1.0 for value in local_barycentric_point])
    
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
        
        # Generates a random point in the triangle by using the barycentric coordinate system.
        v0, v1, v2 = self._perimeter_points

        lambdas = generator.uniform(size=3)
        lambdas = lambdas/sum(lambdas)
        
        return self.translate_to_global(lambdas[0]*v0 + lambdas[1]*v1 + lambdas[2]*v2)

    def get_closest_point(self, local_point: Point) -> Point:
        # Computes the offset between the used triangle center and its incentre.
        bisected_angle_rad = atan(2.0*self.height/self.base)/2.0
        offset = Point((self.height - self.base*tan(bisected_angle_rad))/2.0, 0.0)

        # Checks the orientation of the circle's center compared to the triangle's incentre.
        center_orientation = (local_point + offset).orientation()

        # Chooses which edge to consider based on the center's orientation.
        bisected_angle_degrees = degrees(bisected_angle_rad)
        if center_orientation <= bisected_angle_degrees + 90.0:
            v0, v1 = self._perimeter_points[2], self._perimeter_points[0]
        elif center_orientation >= 270.0 - bisected_angle_degrees:
            v0, v1 = self._perimeter_points[0], self._perimeter_points[1]
        else:
            v0, v1 = self._perimeter_points[1], self._perimeter_points[2]

        # Computes the projection of the vector v0 → local_circle center on v0 → v1.
        point_vector = local_point - v0
        edge_vector = v1 - v0
        
        dot_product = point_vector.x*edge_vector.x + point_vector.y*edge_vector.y

        # Picks the closest point to the circle from the triangle's edge.
        k = max(0.0, min(dot_product/edge_vector.squared_norm(), 1.0))
        closest_point = k * edge_vector + v0

        return closest_point
    
    def get_edge_normal_vector(self, local_point):
        for edge, normal_vector, reference_vector in zip(self._edges, self._edge_normal_vectors, self._edge_reference_vectors):
            if (min(edge[0].x, edge[1].x) - TOLERANCE <= local_point.x <= max(edge[0].x, edge[1].x) + TOLERANCE
            and min(edge[0].y, edge[1].y) - TOLERANCE <= local_point.y <= max(edge[0].y, edge[1].y) + TOLERANCE):
                if local_point.projection(normal_vector) == reference_vector:
                    return normal_vector
        else:
            raise EdgeError("Given point is not on this shape's perimeter. It can't be associated to any edges.")