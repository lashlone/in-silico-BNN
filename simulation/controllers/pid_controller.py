"""
PIDController class module. Inherits from the Controller class. Used to give a more natural movement pattern to the opposing paddle in the simulation.
"""

from simulation.controllers.base_controller import Controller
from simulation.elements.base_element import Element
from simulation.elements.paddle import Paddle
from simulation.geometry.point import Point

class PIDController(Controller):
    """Base class for PIDController objects."""
    kp: float
    ki: float
    kd: float
    _cumulative_error: float
    _last_error: float | None

    def __init__(self, kp: float, ki: float, kd: float):
        """Base class for PIDController objects.
            - kp: Floating value representing the proportional coefficient of the PID controller.
            - ki: Floating value representing the integrative coefficient of the PID controller.
            - kd: Floating value representing the derivative coefficient of the PID controller."""
        
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self._cumulative_error = 0.0
        self._last_error = None

class VerticalPositionPIDController(PIDController):
    """Controls the element's vertical position based on a reference element."""
    reference: Element

    def __init__(self, kp: float, ki: float, kd: float, reference: Element):
        """PIDController that controls the element's vertical position based on a reference element.
            - kp: Floating value representing the proportional coefficient of the PID controller.
            - ki: Floating value representing the integrative coefficient of the PID controller.
            - kd: Floating value representing the derivative coefficient of the PID controller.
            - reference: Element object whose center point is taken as the reference value while computing the error."""
        
        super().__init__(kp, ki, kd)

        if not isinstance(reference, Element):
            raise TypeError(f"unsupported parameter type(s) for reference: '{type(reference).__name__}'")
        
        self.reference = reference
        
    def update(self, controlled_element: Paddle) -> None:
        super().update(controlled_element)

        # Computes the error between the reference and the signal
        error = self.reference.shape.center.y - controlled_element.shape.center.y

        # Handles the integration part of the controller
        self._cumulative_error += error

        # Handles the differential part of the controller
        if self._last_error is None:
            differential_error = 0.0
        else:
            differential_error = error - self._last_error
        self._last_error = error

        # Computes the correction and applies it to the controlled element
        correction = self.kp*error + self.ki*self._cumulative_error + self.kd*differential_error
        controlled_element.set_state(speed=Point(0.0, correction))
    

    
