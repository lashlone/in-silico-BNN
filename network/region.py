"""
Base classes module. 
"""

from network.exceptions import NetworkCommunicationError

import numpy as np
from numpy.typing import NDArray

class Region:
    """Base Class for all regions objects."""
    name: str
    size: int
    internal: bool
    _state: NDArray[np.float16]
    _neurons_index: list[int]

    def __init__(self, name: str, size: int, internal: bool):
        """Base Class for all regions objects."""
        self.name = str(name)
        self.size = int(size)
        if not size > 0:
            raise ValueError("region's size should be bigger then 0.")
        self.internal = bool(internal)

        self._state = np.zeros((size,), dtype=np.float16)

    def set_state(self, state: list[float]) -> None:
        """Set the state of the region according to the given list. List's length must match region's size."""
        if not isinstance(state, list):
            raise TypeError(f"unsupported parameter type(s) for state: '{type(state).__name__}'")
        if not len(state) == self.size:
            raise NetworkCommunicationError(f"given state's length ({len(state)}) does not match region size ({self.size})", faulty_regions=self.name)
        self._state = np.array(state)

    def set_neurons_index(self, first_neuron_index: int) -> None:
        """Set the index of the region's neurons according to the first index."""
        if not 0 <= int(first_neuron_index):
            raise ValueError(f"first index ({first_neuron_index}) should be bigger or equal to zero.")
        self._neurons_index = [first_neuron_index + i for i in range(self.size)]

    def get_state(self) -> NDArray[np.float16]:
        return self._state
    
    def get_neurons_index(self) -> list[int]:
        return self._neurons_index

    def get_indexed_state(self) -> list[tuple[int, float]]:
        """Returns a list of tuple representing the neuron's index and state respectively from the region."""
        return zip(self._neurons_index, self._state)
        
    def is_internal(self) -> bool:
        return self.internal


