"""
PIDController class module. Inherits from the Controller class. Used to give a more natural movement pattern to the opposing paddle in the simulation.
"""

from simulation.controllers.base_controller import Controller
from simulation.elements.base_element import Element
from simulation.geometry.point import Point

class PIDController(Controller):
    """Base class for PIDController object."""
    kp: float
    ki: float
    kd: float
    cumulative_error: float
    last_error: float | None

    def __init__(self, kp: float, ki: float, kd: float, reference: Element):
        """Base class for PIDController object."""
        if not isinstance(reference, Element):
            raise TypeError(f"unsupported parameter type(s) for reference: '{type(reference).__name__}'")
        
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.reference = reference

        self.cumulative_error = 0.0
        self.last_error = None

class VerticalPositionPIDController(PIDController):
    """Creates a VerticalPIDController object that controls the element's vertical position based on a reference element."""

    def update(self, controlled_element):
        super().update(controlled_element)

        # Computes the error between the reference and the signal
        error = self.reference.shape.center.y - controlled_element.shape.center.y

        # Handles the integration part of the controller
        self.cumulative_error += error

        # Handles the differential part of the controller
        if self.last_error is None:
            differential_error = 0.0
        else:
            differential_error = error - self.last_error
        self.last_error = error

        # Computes the correction and applies it to the controlled element
        correction = self.kp*error + self.ki*self.cumulative_error + self.kd*differential_error
        controlled_element.shape.move_center(Point(0.0, correction))
    

    
