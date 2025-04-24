"""
RandomController class module. Inherits from the Controller class. Used to compare the network's result.
"""

from simulation.controllers.base_controller import Controller
from simulation.controllers.exceptions import ControllerInitializationError
from simulation.elements.base_element import Element
from simulation.elements.paddle import Paddle
from simulation.geometry.point import Point

from numpy.random import Generator

class RandomController(Controller):
    """Base class for RandomController objects."""
    _generator: Generator | None

    def __init__(self):
        """Base class for RandomController objects."""
        self._generator = None
    
    def update(self, controlled_element: Element):
        if self._generator is None:
            raise ControllerInitializationError("generator attribute must be set before using this object")
        
        super().update(controlled_element)

        
    def set_generator(self, generator: Generator):
        if not isinstance(generator, Generator):
            raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
        self._generator = generator
        
    
class LinearRandomWalker(RandomController):
    """Controls the element's vertical position based on a linear random walker model."""
    reference_speed: Point
    reference_probabilities: list[float]
    _random_walker_movements_: list[int] = [1, 0, -1]

    def __init__(self, reference_speed: Point, reference_probabilities: list[float] = [1.0/3.0, 1.0/3.0, 1.0/3.0]):
        """RandomController that controls the element's vertical position based on a linear random walker model.
            - reference_speed : Point object representing the random walker step.
            - reference_probability (optional): List of floating values representing the probability that the random walker moves up, stands still or moves down respectively."""
        if not isinstance(reference_speed, Point):            
            raise TypeError(f"unsupported parameter type(s) for reference_speed: '{type(reference_speed).__name__}'")
        if not isinstance(reference_probabilities, list):
            raise TypeError(f"unsupported parameter type(s) for reference_probabilities: '{type(reference_probabilities).__name__}'")
        if not len(reference_probabilities) == 3:
            raise ValueError(f"reference_probabilities size ({len(reference_probabilities)}) does not match the random walker possible movements ({len(self._random_walker_movements_)}).")
        if not all(isinstance(element, float) for element in reference_probabilities):
            raise TypeError("unsupported element type(s) for reference_probabilities element")
        
        super().__init__()
        
        self.reference_speed = reference_speed
        self.reference_probabilities = reference_probabilities

    def update(self, controlled_element: Paddle) -> None:
        super().update(controlled_element)

        walker_choice = self._generator.choice(self._random_walker_movements_, p=self.reference_probabilities)
        
        random_speed = walker_choice * self.reference_speed
        controlled_element.set_state(speed=random_speed)
    

    
