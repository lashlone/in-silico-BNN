"""
Base class module. 

The Element class defined here should not directly be used as an element of a simulation and could make the simulation fail.  
"""

from __future__ import annotations

from simulation.geometry.shape import Shape
from simulation.geometry.vector import Vector2D

class Element:
    """Base class for all Element objects."""
    shape: Shape
    speed: Vector2D
    acceleration: Vector2D

    def __init__(self, shape: Shape, speed: Vector2D = Vector2D(0.0, 0.0), acceleration: Vector2D = Vector2D(0.0, 0.0)):
        """Base class for all Element objects.
            - shape: Shape object representing the shape of the element.
            - speed (optional): Point object representing the speed of the element.
            - acceleration (optional): Point object representing the acceleration of the element."""
        
        if not isinstance(shape, Shape):
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        if not isinstance(speed, Vector2D):
            raise TypeError(f"unsupported parameter type(s) for speed: '{type(speed).__name__}'")
        if not isinstance(acceleration, Vector2D):
            raise TypeError(f"unsupported parameter type(s) for acceleration: '{type(acceleration).__name__}'")

        self.shape = shape
        self.speed = speed
        self.acceleration = acceleration

    def __eq__(self, other) -> bool:
        """Checks if two Element objects are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False

    def __repr__(self) -> str:
        """Element object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Element object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"

    def set_state(self, position: Vector2D | None = None, speed: Vector2D | None = None, acceleration: Vector2D | None = None) -> Element:
        """Set the element's state to the given values. If a parameter is set to None, the element keeps its previous value for this parameter."""
        if position is not None:
            if not isinstance(position, Vector2D):
                raise TypeError(f"unsupported parameter type(s) for position: '{type(position).__name__}'")
            else:
                self.shape.center = position

        if speed is not None:
            if not isinstance(speed, Vector2D):
                raise TypeError(f"unsupported parameter type(s) for speed: '{type(speed).__name__}'")
            else:
                self.speed = speed

        if acceleration is not None:
            if not isinstance(acceleration, Vector2D):
                raise TypeError(f"unsupported parameter type(s) for acceleration: '{type(acceleration).__name__}'")
            else:
                self.acceleration = acceleration

        return self

    def update(self) -> None:
        """Updates the element's state based on its current state."""
        self.shape.move_center(self.speed)
        self.speed += self.acceleration

    def collides_with(self, other: Element) -> bool:
        """Checks for collision between elements."""
        if not isinstance(other, Element):
            raise TypeError(f"unsupported parameter type(s) for other: '{type(other).__name__}'")
        return self.shape.collides_with(other.shape)
    
    def get_position(self) -> Vector2D:
        """Returns the element's position."""
        return self.shape.center