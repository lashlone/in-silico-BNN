"""
Ball class module. Inherits from the Element class. Used to model a moving ball in the simulation.
"""

from __future__ import annotations

from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.vector import Vector2D

class Ball(Element):
    """Simple ball object that follows an UARM and processes collisions."""
    shape: Circle
    speed_range: tuple[float, float]
    
    def __init__(self, shape: Circle, speed: Vector2D, speed_range: tuple[float, float], acceleration: Vector2D):
        """Creates a simple ball object that follows the UARM and processes collisions. This element only allows for circular shapes. 
        A speed range tuple, ('min', 'max'), should be specified to avoid it from potentially breaking the simulation.
            - shape: Circle object representing the shape of the ball.
            - speed: Point object representing the speed of the ball.
            - speed_range: Tuple of floating values representing the range of values that the ball speed's norm can take.
            - acceleration: Point object representing the acceleration of the ball."""
        
        if not isinstance(shape, Circle):
            raise TypeError(f"unsupported parameter type(s) for shape: '{type(shape).__name__}'")
        if not isinstance(speed_range, tuple):
            raise TypeError(f"unsupported parameter type(s) for speed_range: '{type(speed_range).__name__}'")
        else:
            min_speed, max_speed = speed_range
            if min_speed > max_speed:
                raise ValueError("the minimum speed should be smaller than its maximum.")
        
        super().__init__(shape, speed, acceleration)
        self.speed_range = speed_range

    def adjust_speed(self) -> None:
        """Adjusts the ball's speed based on its speed range."""
        min_speed, max_speed = self.speed_range
        current_speed = self.speed.norm()

        if current_speed < min_speed:
            self.speed = (min_speed/current_speed) * self.speed
        elif current_speed > max_speed:
            self.speed = (max_speed/current_speed) * self.speed

    def set_state(self, position = None, speed = None, acceleration = None) -> Ball:
        if speed is not None:
            min_speed, max_speed = self.speed_range            
            if not (min_speed < speed.norm() < max_speed):
                raise ValueError("given speed value's norm is out of bound")
    
        return super().set_state(position, speed, acceleration)

    def update(self) -> None:
        super().update()
        self.adjust_speed()