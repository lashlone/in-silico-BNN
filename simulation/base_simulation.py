"""
Base class module.

The class defined here should not directly be used as Simulation object.
"""

import json
import os

from datetime import datetime
from numpy.random import Generator, PCG64
from numpy.typing import NDArray

from simulation.exceptions import LoadingError
from simulation.elements.base_element import Element

# Imports needed for eval when loading files
from numpy import array                                     # noqa: F401
from simulation.geometry.circle import Circle               # noqa: F401
from simulation.geometry.rectangle import Rectangle         # noqa: F401
from simulation.geometry.point import Point                 # noqa: F401
from simulation.geometry.triangle import IsoscelesTriangle  # noqa: F401

RESULT_PATH_DIR = os.path.join("results")

class Simulation():
    """Base class for all Simulation objects."""
    height: int
    width: int
    frequency: int
    simulation_name: str
    generator_seed: int | None
    _elements: list[Element]
    _simulation_dir_: str
    _generator_: Generator
    _env_history_: list[tuple[str | int | Element | Point]]
    _timer_: int
    
    def __init__(self, height: int, width: int, frequency: int, elements: list[Element], simulation_name: str | None = None, generator_seed: int | None = None):
        """Base class for all Simulation objects.
            - height: Integer representing the simulation height, in pixels.
            - width: Integer representing the simulation width, in pixels.
            - frequency: Integer representing the abstract frequency of the simulation's iteration process, in Hz.
            - elements: list of Element objects representing the elements in the simulation.
            - simulation_name (optional): String representing the simulation's name in the result repository. The default value is {self.__class__.__name__}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}.
            - generator_seed (optional): Integer representing the seed used when creating the simulation's random number generator."""        
        
        self.height = int(height)
        self.width = int(width)
        self.frequency = int(frequency)
        self._elements = elements

        if simulation_name is not None:
            self.simulation_name = str(simulation_name)
        else:
            self.simulation_name = f"{self.__class__.__name__}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}"

        self.generator_seed = generator_seed

        self._simulation_dir_ = os.path.join(RESULT_PATH_DIR, self.simulation_name)
        os.makedirs(self._simulation_dir_, exist_ok=True)

        self._generator_ = Generator(PCG64(generator_seed))
        
        self._env_history_ = [tuple(element.shape for element in self._elements)]
        self._env_history_.append(tuple(element.shape.center for element in self._elements))
        self._timer_ = 0

    def __eq__(self, other) -> bool:
        """Checks if two Simulation objects are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False

    def __repr__(self) -> str:
        """Simulation object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

        # The base simulation must also include its _elements attribute in its representation, but its children must not.
        if issubclass(Simulation, self.__class__):
            filtered_attributes['elements'] = self._elements

        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Simulation object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"

    def step(self) -> None:
        """Updates the states of the simulation's elements based on its previous states, then resolves elements interaction."""
        for element in self._elements:
            element.update()
        self._env_history_.append(tuple(element.shape.center for element in self._elements))
        self._timer_ += 1

    def save_config(self) -> None:
        """Saves the simulation's configuration as a json file."""
        config_file_path = os.path.join(self._simulation_dir_, "config.json")
        with open(config_file_path, "w") as config_file:
            json.dump(repr(self), config_file)
    
    def save_env_history(self, env_history_file_name: str = "env_history.json") -> None:
        """Saves the simulation's environnement history as a json file."""
        env_history_file_path = os.path.join(self._simulation_dir_, env_history_file_name)
        with open(env_history_file_path, "w") as env_history_file:
            json.dump(repr(self._env_history_), env_history_file)

    def get_env_history(self) -> list[tuple[str | int | Element | Point]]:
        return self._env_history_.copy()
    
    def get_simulation_dir(self) -> str:
        return self._simulation_dir_
    
    def get_time(self) -> int:
        return self._timer_

def load_simulation(simulation_name: str) -> Simulation:
    """Loads a simulation from the result directory by its name."""
    simulation_dir = os.path.join(RESULT_PATH_DIR, simulation_name)
    if not os.path.exists(simulation_dir):
        raise FileNotFoundError(f"simulation's directory ({simulation_dir}) was not found in the result repository.")
    
    config_file_path = os.path.join(simulation_dir, "config.json")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"simulation's directory ({simulation_dir}) does not contain a config file.")
    
    with open(config_file_path, "r") as config_file:
        loaded_simulation_data = json.load(config_file)
    loaded_simulation = eval(loaded_simulation_data)

    # Checks the format of the resulting object.
    if not isinstance(loaded_simulation, Simulation):
        raise LoadingError(f"unexpected type when loading the configuration file: '{type(loaded_simulation).__name__}'")
    if not loaded_simulation._simulation_dir_ == simulation_dir:
        raise LoadingError(f"saved simulation's name ({loaded_simulation.simulation_name}) does not match its repository name ({simulation_name})")
    return loaded_simulation
        
def load_env_history(env_history_file_path: str) -> list[tuple[str|int|Element|Point]]:
    """Loads a env_history file into a tuple."""
    with open(env_history_file_path, "r") as env_history_file:
        loaded_env_history_data = json.load(env_history_file)

    loaded_env_history = eval(loaded_env_history_data)

    return loaded_env_history

def load_success_history(success_history_file_path: str) -> list[NDArray]:
    """Loads a success_history file into a list of arrays"""
    with open(success_history_file_path, "r") as success_history_file:
        loaded_success_history_data = json.load(success_history_file)

    loaded_success_history = eval(loaded_success_history_data)

    return loaded_success_history