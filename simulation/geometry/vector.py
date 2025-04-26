"""
Point class module.
"""

from __future__ import annotations
from math import atan, cos, degrees, radians, sin, sqrt

from simulation.geometry.constants import TOLERANCE

class Vector2D:
    """Defines a 2D vector."""
    x: float
    y: float

    def __init__(self, x: float, y: float):
        """Defines a 2D vector."""
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other) -> bool:
        """Two vectors are defined equal if they are relatively close one from another."""
        if isinstance(other, self.__class__):
            return (self - other).squared_norm() <= TOLERANCE ** 2.0
        else:
            return False

    def __ne__(self, other) -> Vector2D:
        return not self.__eq__(other)

    def __add__(self, other) -> Vector2D:
        """Follows the definition of addition between two vectors."""
        if isinstance(other, Vector2D):
            x = self.x + other.x
            y = self.y + other.y
            return Vector2D(x, y)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    def __neg__(self) -> Vector2D:
        """Follows the definition of opposite for a vector."""
        return Vector2D(-self.x, -self.y)
    
    def __sub__(self, other) -> Vector2D:
        """Follows the definition of subtraction between two vectors."""
        if isinstance(other, Vector2D):
            x = self.x - other.x
            y = self.y - other.y
            return Vector2D(x, y)
        else:
            raise TypeError(f"unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        
    def __mul__(self, other: Vector2D) -> float:
        """Follows the definition of the dot product between two vectors."""
        if isinstance(other, Vector2D):
            return self.x * other.x + self.y * other.y
        elif isinstance(other, (int, float)):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'\nSwitch the position of the Point and scalar objects for scalar multiplication.")
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")

    def __rmul__(self, other: int | float) -> Vector2D:
        """Defined by the scalar multiplication of a vector."""
        return Vector2D(float(other)*self.x, float(other)*self.y)
    
    def __repr__(self) -> str:
        """Vector2D object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Vector2D object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"

    def pprint(self) -> str:
        """Vector2D object's pretty representation."""
        return f"[{self.x:.4f}, {self.y:.4f}]"
    
    def round(self, digit_number: int) -> Vector2D:
        """Rounds each coordinates of the point to the given number of digits."""
        return Vector2D(round(self.x, digit_number), round(self.y, digit_number))
    
    def rotate(self, angle: float) -> Vector2D:
        """Rotates a vector around the origin by the specified angle (in degrees)."""
        angle = radians(float(angle))
        return Vector2D(self.x*cos(angle) - self.y*sin(angle), self.x*sin(angle) + self.y*cos(angle))
    
    def squared_norm(self) -> float:
        """Returns the squared euclidean norm of the vector."""
        return self.x**2.0 + self.y**2.0
    
    def norm(self) -> float:
        """Returns the euclidean norm of the vector."""
        return sqrt(self.x**2.0 + self.y**2.0)
    
    def orientation(self) -> float:
        """Returns the orientation (between 0 and 360 degrees) of the vector"""
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
    
    def unit_vector(self) -> Vector2D:
        """Returns the unit vector of this Vector2D object."""
        return (1/self.norm()) * self

    def projection(self, other: Vector2D) -> Vector2D:
        """Returns the projection of this Vector2D object on another vector."""
        if isinstance(other, Vector2D):
            return ((self * other)/other.squared_norm()) * other
        else:
            raise TypeError(f"unsupported parameter type(s) for other: '{type(other).__name__}'")
        
    def reflection(self, other: Vector2D) -> Vector2D:
        """Returns the reflection of this Vector2D object over the axis defined by another vector."""
        if isinstance(other, Vector2D):
            return self - 2.0 * self.projection(other)
        else:
            raise TypeError(f"unsupported parameter type(s) for other: '{type(other).__name__}'") 