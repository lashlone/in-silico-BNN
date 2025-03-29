"""
Paddle class module. Inherits from the Element class. Used to model a paddle in the simulation.
"""

from __future__ import annotations

from simulation.elements.base_element import Element
from simulation.controllers.base_controller import Controller
from simulation.geometry.shape import Shape
from simulation.geometry.point import Point

class Paddle(Element):
    """Simple paddle object that moves only on the vertical axis."""
    controller: Controller
    y_range: tuple[float, float]

    def __init__(self, shape: Shape, controller: Controller, y_range: tuple[float, float]):
        """Creates a paddle object, that moves only on the vertical axis. This element supports every type of shapes defined in the geometry module.
            - shape: Shape object representing the shape of the paddle.
            - controller: Controller object representing the controller of the paddle (see the simulation.controller module).
            - y_range: Tuple of floating values representing the range of y values that the paddle's center can take."""
        
        if not isinstance(controller, Controller):
            raise TypeError(f"unsupported parameter type(s) for controller: '{type(controller).__name__}'")
        if not isinstance(y_range, tuple):
            raise TypeError(f"unsupported parameter type(s) for y_range: '{type(y_range).__name__}'")
        min_y, max_y = y_range
        if min_y > max_y:
            raise ValueError("the minimum y value should be smaller than its maximum")
        
        super().__init__(shape)
        self.controller = controller
        self.y_range = y_range

    def adjust_position(self):
        """Adjusts the paddle's position based on its y range"""
        min_y, max_y = self.y_range
        current_y = self.shape.center.y

        if current_y < min_y:
            self.shape.move_center(Point(0.0, min_y - current_y))
            self.speed = Point(0.0, 0.0)
        elif current_y > max_y:
            self.shape.move_center(Point(0.0, max_y - current_y))
            self.speed = Point(0.0, 0.0)
    
    def set_state(self, position = None, speed = None, acceleration = None) -> Paddle:
        if position is not None:
            min_y, max_y = self.y_range
            
            if not (min_y < position.y < max_y):
                raise ValueError("given position y value is out of bound")
    
        return super().set_state(position, speed, acceleration)

    def update(self):
        self.controller.update(self)
        super().update()
        self.adjust_position()


