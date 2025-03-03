"""
Base classes module.

The classes defined here should not directly be used as Simulation object.
"""

import numpy as np

from simulation.elements.base_element import Element

class Simulation():
    """Base class for all Simulation objects."""
    height: int
    width: int
    frequency: int
    elements: list[Element]
    generator: np.random.Generator
    env_history: list[tuple[str]]
    
    def __init__(self, height: int, width: int, frequency: int, elements: list[Element], generator: np.random.Generator | None = None):
        """Base class for all Simulation objects."""
        if generator is not None:
            if not isinstance(generator, np.random.Generator):
                raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
        
        self.height = int(height)
        self.width = int(width)
        self.frequency = int(frequency)
        self.elements = elements

        if generator is not None:
            self.generator = generator
        else:
            self.generator = np.random.default_rng()

        self.env_history = [(repr(element.shape) for element in self.elements)]
        self.env_history.append((repr(element.shape.center) for element in self.elements))

    def step(self) -> None:
        for element in self.elements:
            element.update()
        self.env_history.append((repr(element.shape.center) for element in self.elements))

  