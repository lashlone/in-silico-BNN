"""
Rectangle class module. Inherits from the Shape class.
"""

from simulation.geometry.shape import Shape
from simulation.geometry.point import Point
from simulation.geometry.circle import Circle

class Rectangle(Shape):
    """Creates a rectangular shape based on its center, its width and its height."""

    def __init__(self, center: Point, width: float, height: float, orientation: float = 0.0):
        """
        Creates a rectangular shape based on its center, its width and its height.
            - width: size of the rectangle on the x-axis.
            - height: size of the rectangle on the y-axis.
            - orientation (optional): angle between the shape's local x-axis and the simulation's x-axis. 
        """
        super().__init__(center, orientation)
        self.width = float(width)
        self.height = float(height)

    def contains_point(self, point: Point) -> bool:
        local_point = self.translate_to_local(point)

        return (-self.width/2.0 <= local_point.x <= self.width/2.0 
            and -self.height/2.0 <= local_point.y <= self.height/2.0) 

    def collides_width(self, shape: Shape) -> bool:
        if isinstance(shape, Circle):
            local_circle_center = self.translate_to_local(shape.center)

            # Picks the closest point to the circle from the rectangle's perimeter.
            closest_x = max(-self.width/2.0, min(local_circle_center.x, self.width/2.0))
            closest_y = max(-self.height/2.0, min(local_circle_center.y, self.height/2.0))
            closest_point = Point(closest_x, closest_y)

            # Checks if the distance from the closest point to the circle's center is smaller than its radius.
            return (local_circle_center - closest_point).squared_norm() <= (shape.radius)**2.0
        
        elif isinstance(shape, Shape):
            return (any([self.contains_point(corner) for corner in shape.get_perimeter_corners()]) 
                 or any([shape.contains_point(corner) for corner in self.get_perimeter_corners()]))
        
        else:
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        
    def get_perimeter_corners(self):
        local_corners = [Point(self.width/2.0, self.height/2.0), Point(-self.width/2.0, self.height/2.0),
                        Point(-self.width/2.0, -self.height/2.0), Point(self.width/2.0, -self.height/2.0)]
        
        return [self.translate_to_global(corner) for corner in local_corners]