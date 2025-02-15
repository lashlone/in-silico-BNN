"""
Triangle class module. Inherits from the Shape class.
"""

from math import atan, degrees, tan
from typing import List

from simulation.geometry.shape import Shape
from simulation.geometry.point import Point
from simulation.geometry.circle import Circle

class IsoscelesTriangle(Shape):
    """Creates a isosceles triangular shape based on its center, its base and its height."""

    def __init__(self, center: Point, base: float, height: float, orientation: float = 0.0):
        """
        Creates a isosceles triangular shape based on its center, its base and its height.
            - center: the center's coordinates of the rectangle boxing the triangle object.
            - base: base of the triangle, parallel to the y-axis.
            - height: height of the triangle, parallel to the x-axis.
            - orientation (optional): angle between the shape's local x-axis and the simulation's x-axis. 
        """
        super().__init__(center, orientation)
        self.base = float(base)
        self.height = float(height)
        
        self.reference_vectors = [Point(self.height/2.0, 0.0),
                                  Point(-self.height/2.0, self.base/2.0),
                                  Point(-self.height/2.0, -self.base/2.0)]

    def contains_point(self, point: Point) -> bool:
        local_point = self.translate_to_local(point)
        local_barycentric_point = self.get_barycentric_coordinates(local_point)

        return all([0.0 <= value <= 1.0 for value in local_barycentric_point])
    
    def collides_width(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            local_circle_center = self.translate_to_local(shape.center)

            # Computes the offset between the used triangle center and its incentre.
            bisected_angle_rad = atan(2.0*self.height/self.base)/2.0
            offset = Point((self.height - self.base*tan(bisected_angle_rad))/2.0, 0.0)

            # Checks the orientation of the circle's center compared to the triangle's incentre.
            center_orientation = (local_circle_center + offset).orientation()

            # Chooses which edge to consider based on the center's orientation
            bisected_angle_degrees = degrees(bisected_angle_rad)
            if center_orientation <= bisected_angle_degrees + 90.0:
                v0, v1 = self.reference_vectors[0], self.reference_vectors[1]
            elif center_orientation >= 270.0 - bisected_angle_degrees:
                v0, v1 = self.reference_vectors[0], self.reference_vectors[2]
            else:
                v0, v1 = self.reference_vectors[1], self.reference_vectors[2]

            # Computes the projection of the vector v0 → local_circle center on v0 → v1.
            point_vector = local_circle_center - v0
            edge_vector = v1 - v0
            
            dot_product = point_vector.x*edge_vector.x + point_vector.y*edge_vector.y

            # Picks the closest point to the circle from the triangle's edge.
            k = max(0.0, min(dot_product/edge_vector.squared_norm(), 1.0))
            closest_point = k * edge_vector + v0

            # Checks if the distance from the closest point to the circle's center is smaller than its radius.
            return (local_circle_center - closest_point).squared_norm() <= (shape.radius)**2.0
        
        elif isinstance(shape, Shape):
            return (any([self.contains_point(corner) for corner in shape.get_perimeter_corners()]) 
                 or any([shape.contains_point(corner) for corner in self.get_perimeter_corners()]))
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_corners(self) -> list[Point]:
        local_corners = self.reference_vectors
        
        return [self.translate_to_global(corner) for corner in local_corners]
    
    def get_barycentric_coordinates(self, point: Point) -> List[float]:
        """Returns the barycentric coordinates of the given point, using the triangle's three vertices as base."""
        v0, v1, v2 = self.reference_vectors

        # Defines a linear system with the condition on the sum of the lambdas. See references for more explications.
        a1 = v1 - v0
        a2 = v2 - v0
        b = point - v0

        # Uses Cramer's rule in 2D to quickly solve the linear system.
        det_a = a1.x*a2.y - a1.y*a2.x

        lambda1 = (b.x*a2.y - b.y*a2.x)/(det_a)
        lambda2 = (a1.x*b.y - a1.y*b.x)/(det_a)

        # Solves for lambda zero with the condition on the sum of the lambdas.
        lambda0 = 1.0 - lambda1 - lambda2

        return lambda0, lambda1, lambda2
    