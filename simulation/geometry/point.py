"""
Point class module.
"""

from __future__ import annotations
from math import atan, cos, degrees, radians, sin, sqrt

TOLERANCE = 1e-8

class Point:
    """Defines a point as a 2D vector."""
    x: float
    y: float

    def __init__(self, x: float, y: float):
        """Defines a point as a 2D vector."""
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return (self - other).squared_norm() <= TOLERANCE ** 2.0
        else:
            return False

    def __ne__(self, other) -> Point:
        return not self.__eq__(other)

    def __add__(self, other) -> Point:
        if isinstance(other, Point):
            x = self.x + other.x
            y = self.y + other.y
            return Point(x, y)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __neg__(self) -> Point:
        return Point(-self.x, -self.y)
    
    def __sub__(self, other) -> Point:
        if isinstance(other, Point):
            x = self.x - other.x
            y = self.y - other.y
            return Point(x, y)
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        
    def __rmul__(self, other: int | float) -> Point:
        return Point(float(other)*self.x, float(other)*self.y)
    
    def __repr__(self) -> str:
        """Object's representation for testing purposes."""
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def __str__(self) -> str:
        """Object's clean representation."""
        return f"[{self.x:.4f}, {self.y:.4f}]"
    
    def round(self, digit_number: int) -> Point:
        """Rounds each coordinates of the point to the given number of digits."""
        return Point(round(self.x, digit_number), round(self.y, digit_number))
    
    def rotate(self, angle: float) -> Point:
        """Rotates a point around the origin by the specified angle (in degrees)."""
        angle = radians(float(angle))
        return Point(self.x*cos(angle) - self.y*sin(angle), self.x*sin(angle) + self.y*cos(angle))
    
    def squared_norm(self) -> float:
        """Returns the squared euclidean norm of the point when considered as a vector."""
        return self.x**2.0 + self.y**2.0
    
    def norm(self) -> float:
        """Returns the euclidean norm of the point when considered as a vector."""
        return sqrt(self.x**2.0 + self.y**2.0)
    
    def orientation(self) -> float:
        """Returns the orientation (between 0 and 360 degrees) of the point when considered as a vector."""

        # Checks if the vector is vertical.
        if self.x == 0.0:
            angle = 90.0 if self.y > 0.0 else 0.0 if self.y == 0.0 else -90.0
        else:
            angle = degrees(atan(self.y/self.x))

            # Adjusts the angle if the x coordinate is negative.
            if self.x < 0.0:
                angle += 180.0

        # Adjusts the angle to be in the specified range.
        if angle < 0.0:
            angle += 360.0

        return angle