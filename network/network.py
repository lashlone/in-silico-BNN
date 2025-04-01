"""
Network class module.
"""

from __future__ import annotations
from typing import Callable
from numpy.typing import NDArray

from network.exceptions import NetworkCommunicationError, NetworkInitializationError
from network.regions import Region

import numpy as np
from collections import deque

class Network():
    regions: list[Region]
    regions_connectome: dict[str, dict[str, Callable[[int, int], NDArray[np.float32]]]]
    recovery_state_energy_ratio: float
    state_history_size: int
    decay_coefficient: float
    exploration_rate: float
    strengthening_exponent: float
    reward_fn_period: int
    punish_fn_period: int 
    punish_fn_min_signal_period: int
    punish_fn_max_signal_period: int
    k_value: float
    _size_: int
    _sensory_regions_names_ : list[str]
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
                    state_history_size: int = 12,
                    decay_coefficient: float = 0.02,
                    exploration_rate: float = 0.001,
                    strengthening_exponent: float = 1.1,
                    reward_fn_period: int = 4,
                    punish_fn_period: int = 16,
                    punish_fn_min_signal_period: int = 4,
                    punish_fn_max_signal_period: int = 8,
                    k_value: float = 1,
                ):
        
        if not isinstance(regions, list):
            raise TypeError(f"unsupported parameter type(s) for regions: '{type(regions).__name__}'")
        if regions == []:
            raise ValueError("regions can not be empty.")
        known_regions = set()
        for region in regions:
            if not isinstance(region, Region):
                raise TypeError(f"unsupported element type(s) for regions: '{type(region).__name__}'")
            if region.name in known_regions:
                raise NetworkInitializationError("region's name in the network must be unique.")
            known_regions.add(region.name)
        if not isinstance(regions_connectome, dict):
            raise TypeError(f"unsupported parameter type(s) for region_connectome: '{type(regions_connectome).__name__}'")
        referenced_regions = set([str(region) for region in regions_connectome.keys()])
        referenced_regions.union(set([str(region) for region_list in regions_connectome.values() for region in region_list.keys()]))
        unknown_regions = referenced_regions - known_regions
        if not unknown_regions == set():
            raise ValueError(f"unknown regions where used in the network's connectome's definition: {unknown_regions}")

        self.regions = regions
        self.regions_connectome = regions_connectome
        self.recovery_state_energy_ratio = float(recovery_state_energy_ratio)
        self.state_history_size = int(state_history_size)
        self.decay_coefficient = float(decay_coefficient)
        self.exploration_rate = float(exploration_rate)
        self.strengthening_exponent = float(strengthening_exponent)
        self.reward_fn_period = int(reward_fn_period)
        self.punish_fn_period = int(punish_fn_period)
        self.punish_fn_min_signal_period = float(punish_fn_min_signal_period)
        self.punish_fn_max_signal_period = float(punish_fn_max_signal_period)
        self.k_value = float(k_value)

        sensory_regions_names = []
        internal_regions_indexes = []
        current_index = 0
        for region in self.regions:
            region.set_neurons_index(current_index)
            if region.is_internal():
                internal_regions_indexes += region.get_neurons_index()
            else:
                sensory_regions_names.append(region.name)
            current_index += region.size

        self._size_ = current_index
        self._sensory_regions_names_ = sensory_regions_names
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
        internal_state = self.get_internal_state()
        internal_triggered_neurons = np.array(internal_state == 1.0).astype(np.float16)
        internal_non_triggered_neurons = np.array(internal_state != 1.0).astype(np.float16)
        internal_probability_matrix = np.nan_to_num(self.get_internal_conformation(), copy=True, nan=1.0)
        internal_resting_probability_vector = np.prod(internal_probability_matrix ** internal_triggered_neurons, axis=1)
        safe_internal_resting_prob_vector = np.where(internal_resting_probability_vector > 0, internal_resting_probability_vector, 1)
        internal_triggering_probability_vector = 1.0 - internal_resting_probability_vector
        safe_internal_triggering_prob_vector = np.where(internal_triggering_probability_vector > 0, internal_triggering_probability_vector, 1)
        internal_entropy_vector = -safe_internal_resting_prob_vector * np.log2(safe_internal_resting_prob_vector) - safe_internal_triggering_prob_vector * np.log2(safe_internal_triggering_prob_vector)
        network_internal_entropy = internal_entropy_vector @ internal_non_triggered_neurons

        state = self.get_state()
        triggered_neurons = np.array(state == 1.0).astype(np.float16)
        non_triggered_neurons = np.array(state != 1.0).astype(np.float16)
        probability_matrix = np.nan_to_num(self.get_conformation(), copy=True, nan=1.0)
        resting_probability_vector = np.prod(probability_matrix ** triggered_neurons, axis=1)
        safe_resting_prob_vector = np.where(resting_probability_vector > 0, resting_probability_vector, 1)
        triggering_probability_vector = 1.0 - resting_probability_vector
        safe_triggering_prob_vector = np.where(triggering_probability_vector > 0, triggering_probability_vector, 1)
        entropy_vector = -safe_resting_prob_vector * np.log2(safe_resting_prob_vector) - safe_triggering_prob_vector * np.log2(safe_triggering_prob_vector)
        network_global_entropy = entropy_vector @ non_triggered_neurons

        network_potential_energy = -sum(internal_state)

        free_energy = (network_internal_entropy - network_global_entropy) - self.k_value * network_potential_energy
        self._free_energy_history_.append(free_energy)

        return free_energy

    def propagate_signal(self, generator: np.random.Generator, sensory_signal: dict[str, list[float]] | None = None) -> None:
        """This method propagates the signal in the network. If given, the sensory_signal represents the sensory signal perceived by the agent from the environnement in the form of a dictionary."""
        self._state = np.concatenate([region.get_state() for region in self.regions])

        # Applies sensory_signal
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

        triggered_neurons = np.array(self._state == 1.0).astype(np.float16)
        probability_matrix = np.nan_to_num(self._conformation, copy=True, nan=1.0)
        triggering_probability_matrix = 1.0 - np.prod(probability_matrix ** triggered_neurons, axis=1)

        for region in self.regions:
            if region.name not in sensory_signal:
                updated_state = []
                for neuron_index, neuron_state in region.get_indexed_state():
                    # Triggered neurons simply go back to the recovery state
                    if neuron_state == 1.0:
                        updated_state.append(self.recovery_state_energy_ratio)

                    # Recovering neurons go back to the rest state if they are not retriggered.
                    elif neuron_state == self.recovery_state_energy_ratio:
                        if generator.uniform() <= triggering_probability_matrix[neuron_index]:
                            updated_state.append(self.recovery_state_energy_ratio)
                        else:
                            updated_state.append(0.0)
                    
                    # Resting neurons checks if they should trigger based on their neighbors' activity and the network's conformation
                    elif neuron_state == 0.0:
                        if generator.uniform() <= triggering_probability_matrix[neuron_index]:
                            updated_state.append(1.0)
                        else:
                            updated_state.append(0.0)
                region.set_state(updated_state)
        self._state_history_.append(self.get_state())

    def optimize_connections(self) -> None:
        internal_state = self.get_internal_state()
        last_internal_state = self.get_last_internal_state()
        internal_conformation = self.get_internal_conformation()

        internal_conformation = self.decay_coefficient + (1 - self.decay_coefficient) * internal_conformation
        for neuron_index, neuron_state in enumerate(internal_state):
            if neuron_state == 1.0:
                for neighbor_index, neighbor_state in enumerate(last_internal_state):
                    if neighbor_state == 1.0:
                        internal_conformation[neuron_index, neighbor_index] = internal_conformation[neuron_index, neighbor_index] ** self.strengthening_exponent
                    else:
                        internal_conformation[neighbor_index, neuron_index] = internal_conformation[neighbor_index, neuron_index] * (1 - self.exploration_rate)

        self._conformation[np.ix_(self._internal_regions_indexes_, self._internal_regions_indexes_)] = internal_conformation

    def reward(self):
        internal_conformation = self.get_internal_conformation()

        for _ in range(self.reward_fn_period):
            internal_conformation = self.decay_coefficient + (1 - self.decay_coefficient) * internal_conformation
            internal_conformation = internal_conformation ** self.strengthening_exponent
        
        self._conformation[np.ix_(self._internal_regions_indexes_, self._internal_regions_indexes_)] = internal_conformation

    def punish(self, generator: np.random.Generator):
        sensory_region_periods = generator.integers(low=self.punish_fn_min_signal_period, high=self.punish_fn_max_signal_period, size=(len(self._sensory_regions_names_),))
        sensory_region_delays = generator.integers(low=0, high=self.punish_fn_period//2, size=(len(self._sensory_regions_names_),))
        for i in range(self.punish_fn_period):
            sensory_signal = dict()
            for period, delay, region_name in zip(sensory_region_periods, sensory_region_delays, self._sensory_regions_names_):
                if i < delay:
                    sensory_signal[region_name] = [0.0] * self._region_dict_[region_name].size
                else:
                    if (i - delay) % period == 0:
                        sensory_signal[region_name] = [1.0] * self._region_dict_[region_name].size
                    else:
                        sensory_signal[region_name] = [0.0] * self._region_dict_[region_name].size

            self.propagate_signal(generator=generator, sensory_signal=sensory_signal)
            self.optimize_connections()

    def set_state(self, state: np.ndarray):
        if not isinstance(state, np.ndarray):
            raise TypeError(f"unsupported parameter type(s) for state: '{type(state).__name__}'")
        if not len(state) == self._size_:
            raise ValueError(f"given state array's length ({len(state)}) does not match network's size ({self._size_}).")
        
        for region in self.regions:
            region.set_state(list(state[np.ix_(region.get_neurons_index())]))
        self._state = self.get_state()
        self._state_history_.append(self._state)

    def get_conformation(self):
        return np.copy(self._conformation)
    
    def get_free_energy_history(self):
        return self._free_energy_history_.copy()

    def get_internal_conformation(self):
        return np.copy(self._conformation[np.ix_(self._internal_regions_indexes_, self._internal_regions_indexes_)])
    
    def get_internal_state(self):
        return np.concatenate([region.get_state() for region in self.regions if region.is_internal()])
    
    def get_last_internal_state(self):
        return np.copy(self._state_history_[-2][np.ix_(self._internal_regions_indexes_,)])
    
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
                motor_signal[signal_index] += np.mean(state.astype(np.float64)[list_indexes])
        
        return list(motor_signal / self.state_history_size)
    
    def get_size(self):
        return self._size_
    
    def get_state(self):
        return np.concatenate([region.get_state() for region in self.regions])