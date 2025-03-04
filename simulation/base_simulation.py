"""
Base classes module.

The classes defined here should not directly be used as Simulation object.
"""

import json
import numpy as np
import os
from datetime import datetime

from simulation.elements.base_element import Element

RESULT_PATH_DIR = os.path.join("results")

class Simulation():
    """Base class for all Simulation objects."""
    height: int
    width: int
    frequency: int
    elements: list[Element]
    generator: np.random.Generator
    simulation_name: str
    _env_history: list[tuple[str]]
    _simulation_dir: str
    
    def __init__(self, height: int, width: int, frequency: int, elements: list[Element], generator: np.random.Generator | None = None, simulation_name: str | None = None):
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
        if simulation_name is not None:
            self.simulation_name = str(simulation_name)
        else:
            self.simulation_name = f"{self.__class__.__name__}_{datetime.now().strftime('%d-%m-%Y_%H:%M')}"

        self._simulation_dir = os.path.join(RESULT_PATH_DIR, self.simulation_name)
        os.makedirs(self._simulation_dir, exist_ok=True)
        
        self._env_history = [tuple(element.shape for element in self.elements)]
        self._env_history.append(tuple(element.shape.center for element in self.elements))

    def __repr__(self) -> str:
        """Object's representation for testing purposes."""
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def __str__(self) -> str:
        """Object's string representation, excluding hidden attributes."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in filtered_attributes.items())})"

    def step(self) -> None:
        """Updates the states of all of the simulation's elements based on its previous states, then resolves elements interaction."""
        for element in self.elements:
            element.update()
        self._env_history.append(tuple(element.shape.center for element in self.elements))

    def save_env_history_file(self):
        """Saves the simulation's environnement history as a json file."""
        env_history_file_path = os.path.join(self._simulation_dir, "env_history.json")
        repr_env_history = [[str(element) for element in elements] for elements in self._env_history]
        with open(env_history_file_path, "w") as env_history_file:
            json.dump(repr_env_history, env_history_file)

    def save_config_file(self):
        pass