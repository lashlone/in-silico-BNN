"""
Network class module.
"""

from __future__ import annotations
from typing import Callable
from numpy.typing import NDArray

from network.exceptions import NetworkCommunicationError, NetworkInitializationError
from network.region import Region

import numpy as np
from collections import deque

class Network():
    regions: list[Region]
    regions_connectome: dict[str, dict[str, Callable[[int, int], NDArray[np.float32]]]]
    recovery_state_energy_ratio: float
    state_history_size: int
    decay_rate: float
    exploration_rate: float
    strengthening_rate: float
    _known_regions_: set[str]
    _internal_regions_indexes_ : list[int]
    _region_dict_: dict[str, Region]
    _state: NDArray[np.float16]
    _state_history_: deque[NDArray[np.float16]]
    _free_energy_history_: list[float]
    _conformation: NDArray[np.float32]

    def __init__(
                    self,
                    regions: list[Region],
                    regions_connectome: dict[str, dict[str, Callable[[int, int], NDArray[np.float32]]]],
                    recovery_state_energy_ratio: float = 0.5,
                    state_history_size: int = 16,
                    decay_rate: float = 0.005,
                    exploration_rate: float = 0.02,
                    strengthening_rate: float = 0.33
                ):
        
        if not isinstance(regions, list):
            raise TypeError(f"unsupported parameter type(s) for regions: '{type(regions).__name__}'")
        if regions == []:
            raise ValueError("regions can not be empty.")
        self._known_regions_ = set()
        for region in regions:
            if not isinstance(region, Region):
                raise TypeError(f"unsupported element type(s) for regions: '{type(region).__name__}'")
            if region.name in self._known_regions_:
                raise NetworkInitializationError("region's name in the network must be unique.")
            self._known_regions_.add(region.name)
        if not isinstance(regions_connectome, dict):
            raise TypeError(f"unsupported parameter type(s) for region_connectome: '{type(regions_connectome).__name__}'")
        referenced_regions = set([str(region) for region in regions_connectome.keys()])
        referenced_regions.union(set([str(region) for region_list in regions_connectome.values() for region in region_list.keys()]))
        unknown_regions = referenced_regions - self._known_regions_
        if not unknown_regions == set():
            raise ValueError(f"unknown regions where used in the network's connectome's definition: {unknown_regions}")

        self.regions = regions
        self.regions_connectome = regions_connectome
        self.recovery_state_energy_ratio = float(recovery_state_energy_ratio)
        self.state_history_size = int(state_history_size)
        self.decay_rate = float(decay_rate)
        self.exploration_rate = float(exploration_rate)
        self.strengthening_rate = float(strengthening_rate)

        internal_regions_indexes = []
        current_index = 0
        for region in self.regions:
            region.set_neurons_index(current_index)
            if region.is_internal():
                internal_regions_indexes += region.get_neurons_index()
            current_index += region.size

        self._internal_regions_indexes_ = internal_regions_indexes
        self._region_dict_ = {region.name: region for region in self.regions}
        self._state = np.concatenate([region.get_state() for region in self.regions])
        self._state_history_ = deque([self._state] * self.state_history_size, maxlen=self.state_history_size)
        self._free_energy_history_ = []
        
        conformation_builder = []
        for source_region in regions:
            source_region_conformation = []
            for target_region in regions:
                if source_region.name in self.regions_connectome.keys() and target_region.name in self.regions_connectome[source_region.name].keys():
                    source_region_conformation.append(1.0 - self.regions_connectome[source_region.name][target_region.name](target_region.size, source_region.size))
                else:
                    source_region_conformation.append(np.full((target_region.size, source_region.size), np.nan))
            conformation_builder.append(np.concatenate(source_region_conformation, axis=0))
        self._conformation = np.concatenate(conformation_builder, axis=1)

    def compute_free_energy(self) -> float:
        """Computes the network's free energy, stores it in its history and returns the computed value"""
        state = self.get_state()
        triggered_neurons = np.array(state == 1.0).astype(np.float16)
        resting_neurons = np.array(state == 0.0).astype(np.float16)
        probability_matrix = np.nan_to_num(self._conformation, copy=True, nan=1.0)
        resting_probability_matrix = np.prod(probability_matrix ** triggered_neurons)
        triggering_probability_matrix = 1.0 - resting_probability_matrix
        entropy_matrix = resting_probability_matrix * np.log(resting_probability_matrix) + triggering_probability_matrix * np.log(triggering_probability_matrix)
        network_global_entropy = entropy_matrix @ resting_neurons

        internal_state = self.get_internal_state()
        internal_triggered_neurons = np.array(internal_state == 1.0).astype(np.float16)
        internal_resting_neurons = np.array(internal_state == 0.0).astype(np.float16)
        internal_probability_matrix = np.nan_to_num(self.get_internal_conformation(), copy=True, nan=1.0)
        internal_resting_probability_matrix = np.prod(internal_probability_matrix ** internal_triggered_neurons)
        internal_triggering_probability_matrix = 1.0 - internal_resting_probability_matrix
        internal_entropy_matrix = internal_resting_probability_matrix * np.log(internal_resting_probability_matrix) + internal_triggering_probability_matrix * np.log(internal_triggering_probability_matrix)
        network_internal_entropy = internal_entropy_matrix @ internal_resting_neurons

        network_potential_energy = sum(internal_state)

        free_energy = (network_internal_entropy - network_global_entropy) + network_potential_energy
        self._free_energy_history_.append(free_energy)

        return free_energy

    def propagate_signal(self, generator: np.random.Generator, sensory_signal: dict[str, list[float]] | None = None):
        """This method propagates the signal in the network. If given, the sensory_signal represents the sensory signal perceived by the agent from the environnement in the form of a dictionary."""
        if sensory_signal is not None:
            faulty_regions = []
            for region_name in sensory_signal:
                try:
                    self._region_dict_[region_name].set_state(sensory_signal[region_name])
                except KeyError:
                    faulty_regions.append(region_name)
            if not faulty_regions == []:
                raise NetworkCommunicationError(f"unknown region '{faulty_regions}'", faulty_regions=faulty_regions)
        else:
            sensory_signal = []
        
        self._state = np.concatenate([region.get_state() for region in self.regions])

        triggered_neurons = np.floor(self._state)
        probability_matrix = np.nan_to_num(self._conformation, copy=True, nan=1.0)
        triggering_probability_matrix = 1.0 - np.prod(probability_matrix ** triggered_neurons, axis=1)

        for region in self.regions:
            if region.name not in sensory_signal:
                updated_state = []
                for neuron_index, neuron_state in region.get_indexed_state():
                    # Triggered neurons simply go back to the recovery state
                    if neuron_state == 1.0:
                        updated_state.append(self.recovery_state_energy_ratio)

                    # Recovering neurons simply go back to the rest state
                    elif neuron_state == self.recovery_state_energy_ratio:
                        updated_state.append(0.0)
                    
                    # Resting neurons checks if they should trigger based on their neighbors' activity and the network's conformation
                    elif neuron_state == 0.0:
                        if generator.uniform() <= triggering_probability_matrix[neuron_index]:
                            updated_state.append(1.0)
                        else:
                            updated_state.append(0.0)
                region.set_state(updated_state)
        self._state_history_.append(self.get_state())

    def optimize_connections(self):
        self._conformation = self.decay_rate + (1 - self.decay_rate) * self._conformation
        for neuron_index, neuron_state in enumerate(self._state):
            if neuron_state == self.recovery_state_energy_ratio:
                for neighbor_index, neighbor_state in enumerate(self._state):
                    if neighbor_state == 1.0:
                        self._conformation[neighbor_index, neuron_index] = self.strengthening_rate * self._conformation[neighbor_index, neuron_index]
                    else:
                        self._conformation[neighbor_index, neuron_index] = max(0.0, self._conformation[neighbor_index, neuron_index] - self.exploration_rate)

    def get_conformation(self):
        return self._conformation
    
    def get_state(self):
        return np.concatenate([region.get_state() for region in self.regions])
    
    def get_internal_conformation(self):
        return self._conformation[np.ix_(self._internal_regions_indexes_, self._internal_regions_indexes_)]
    
    def get_internal_state(self):
        return np.concatenate([region.get_state() for region in self.regions if region.is_internal()])
    
    def get_motor_signal(self, accessed_regions: tuple[str]) -> list[float]:
        accessed_indexes = []
        for region_name in accessed_regions:
            try:
                accessed_indexes.append(self._region_dict_[region_name].get_neurons_index())
            except KeyError:
                raise NetworkCommunicationError(f"unknown region '{region_name}'")
        
        motor_signal = np.zeros(len(accessed_regions))
        for state in self._state_history_:
            for signal_index, list_indexes in enumerate(accessed_indexes):
                motor_signal[signal_index] += np.mean(state[list_indexes])
        
        return motor_signal / self.state_history_size
