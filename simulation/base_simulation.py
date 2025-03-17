"""
Base classes module.

The classes defined here should not directly be used as Simulation object.
"""

import json
from numpy.random import Generator, PCG64
import os
from datetime import datetime

from simulation.exceptions import LoadingError
from simulation.elements.base_element import Element

# Imports needed for eval when loading a simulation
from simulation.geometry.circle import Circle  # noqa: F401
from simulation.geometry.rectangle import Rectangle # noqa: F401
from simulation.geometry.point import Point # noqa: F401
from simulation.geometry.triangle import IsoscelesTriangle # noqa: F401

RESULT_PATH_DIR = os.path.join("results")

class Simulation():
    """Base class for all Simulation objects."""
    height: int
    width: int
    frequency: int
    simulation_name: str
    generator_seed: int | None
    _elements: list[Element]
    _simulation_dir: str
    _generator_: Generator
    _env_history: list[tuple[str | int | Element | Point]]
    
    def __init__(self, height: int, width: int, frequency: int, elements: list[Element], simulation_name: str | None = None, generator_seed: int | None = None):
        """Base class for all Simulation objects."""        
        self.height = int(height)
        self.width = int(width)
        self.frequency = int(frequency)
        self._elements = elements

        if simulation_name is not None:
            self.simulation_name = str(simulation_name)
        else:
            self.simulation_name = f"{self.__class__.__name__}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

        self.generator_seed = generator_seed

        self._simulation_dir = os.path.join(RESULT_PATH_DIR, self.simulation_name)
        os.makedirs(self._simulation_dir, exist_ok=True)

        self._generator_ = Generator(PCG64(generator_seed))
        
        self._env_history = [(self._simulation_dir, self.width, self.height), tuple(element.shape for element in self._elements)]
        self._env_history.append(tuple(element.shape.center for element in self._elements))

    def __eq__(self, other) -> bool:
        """Checks if two Simulation are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False

    def __repr__(self) -> str:
        """Object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

        # The base simulation must also include its _elements attribute in its representation, but its children must not.
        if issubclass(Simulation, self.__class__):
            filtered_attributes['elements'] = self._elements

        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"

    def step(self) -> None:
        """Updates the states of the simulation's elements based on its previous states, then resolves elements interaction."""
        for element in self._elements:
            element.update()
        self._env_history.append(tuple(element.shape.center for element in self._elements))

    def save_config(self) -> None:
        """Saves the simulation's configuration as a json file."""
        config_file_path = os.path.join(self._simulation_dir, "config.json")
        with open(config_file_path, "w") as config_file:
            json.dump(repr(self), config_file)
    
    def save_env_history(self, env_history_file_name: str = "env_history.json") -> None:
        """Saves the simulation's environnement history as a json file."""
        env_history_file_path = os.path.join(self._simulation_dir, env_history_file_name)
        with open(env_history_file_path, "w") as env_history_file:
            json.dump(repr(self._env_history), env_history_file)

def load_simulation(simulation_name: str) -> Simulation:
    """Loads a simulation from the result directory by its name and checks the format of the resulting object."""
    simulation_dir = os.path.join(RESULT_PATH_DIR, simulation_name)
    if not os.path.exists(simulation_dir):
        raise FileNotFoundError(f"simulation's directory ({simulation_dir}) was not found in the result repository.")
    
    config_file_path = os.path.join(simulation_dir, "config.json")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"simulation's directory ({simulation_dir}) does not contain a config file.")
    
    with open(config_file_path, "r") as config_file:
        loaded_simulation_data = json.load(config_file)

    loaded_simulation = eval(loaded_simulation_data)
    if not isinstance(loaded_simulation, Simulation):
        raise LoadingError(f"unexpected type when loading the configuration file: '{type(loaded_simulation).__name__}'")
    if not loaded_simulation._simulation_dir == simulation_dir:
        raise LoadingError(f"saved simulation's name ({loaded_simulation.simulation_name}) does not match its repository name ({simulation_name})")
    return loaded_simulation
        
def load_env_history(env_history_file_path: str) -> list[tuple[str|int|Element|Point]]:
    """Loads a env_history file into a tuple. Does not check the format of the resulting object."""
    with open(env_history_file_path, "r") as env_history_file:
        loaded_env_history_data = json.load(env_history_file)

    loaded_env_history = eval(loaded_env_history_data)

    return loaded_env_history