"""
Graph generation module. This module contains the functions to generate the initial conformation of the network.
"""

from typing import Callable
from numpy.typing import NDArray

import numpy as np
from numpy.random import Generator

def fixed_average_transmission(transmission_average: float, generator: Generator, tolerance: float = 1e-6) -> Callable[[int, int], NDArray]:
    """
    Generator function for fixed average potential of action transmission probability through two regions of a graph.
        - transmission_average : the average probability for the transmission of potential of action between two nods.
        - generator : Generator object to use when generating random numbers.
    """
    if not isinstance(generator, Generator):
        raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
    transmission_average = float(transmission_average)
    if not 0.0 < transmission_average < 1.0:
        raise ValueError(f"The average transmission rate should be between 0.0 and 1.0, not {transmission_average}.")
    
    def graph_generation_fn(target_region_size: int, source_region_size: int) -> NDArray:
        conformation = generator.uniform(size=(target_region_size, source_region_size))
        current_average = conformation.mean(axis=1, keepdims=True)
        while np.any(np.abs(current_average - transmission_average) >= tolerance):
            corrected_conformation = (transmission_average/current_average) * conformation
            clipped_conformation = np.clip(corrected_conformation, 0.0, 1.0)
            current_average = clipped_conformation.mean(axis=1, keepdims=True)
        return clipped_conformation.astype(np.float32)
    return graph_generation_fn

def self_referring_fixed_average_transmission(transmission_average: float, generator: Generator, tolerance: float = 1e-6) -> Callable[[int, int], NDArray]:
    """
    Generator function for fixed average potential of action transmission probability within the same region of a graph.
        - transmission_average : the average probability for the transmission of potential of action between two nods.
        - generator : Generator object to use when generating random numbers.
    """
    if not isinstance(generator, Generator):
        raise TypeError(f"unsupported parameter type(s) for generator: '{type(generator).__name__}'")
    transmission_average = float(transmission_average)
    if not 0.0 < transmission_average < 1.0:
        raise ValueError(f"The average transmission rate should be between 0.0 and 1.0, not {transmission_average}.")
    
    def graph_generation_fn(target_region_size: int, source_region_size: int) -> NDArray:
        if target_region_size != source_region_size:
            raise ValueError("Since the region is referring to itself, both given sizes should be equal.")
        
        conformation = generator.uniform(size=(target_region_size, source_region_size))
        np.fill_diagonal(conformation, np.nan)
        current_average = np.nanmean(conformation, axis=1, keepdims=True)
        while np.any(np.abs(current_average - transmission_average) >= tolerance):
            corrected_conformation = (transmission_average/current_average) * conformation
            clipped_conformation = np.clip(corrected_conformation, 0.0, 1.0)
            current_average = np.nanmean(clipped_conformation, axis=1, keepdims=True)
        return clipped_conformation.astype(np.float32)
    return graph_generation_fn