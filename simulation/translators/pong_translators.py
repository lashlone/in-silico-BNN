
from __future__ import annotations

from simulation.pong import Pong

class PongSignalTranslator:
    """This class allows the creation of a sensory signal from a Pong simulation object."""
    regions_name: list[str]
    min_frequency: int
    max_frequency: int
    _nb_topographic_regions: int
    _neurons_by_regions: int
    _topographic_region_length: float
    _timer_: int
    _simulation: Pong

    def __init__(self, regions_name: list[str], region_size: int, min_frequency: int, max_frequency: int):
        """This class allows the creation of a sensory signal from a Pong simulation object.
            - region_name: name of the associated signal region in the network.
            - region_size: size of the associated signal region in the network.
            - min_frequency: minimum frequency of the signal, when the ball is farthest from the agent.
            - max_frequency: maximum frequency of the signal, when the ball is closest to the agent.
            - nb_topographic_regions: number of topographic regions in the associated signal region. Defines how to split the vertical grid into regions.

        This class's simulation attribute has to be defined through its setter before the generate_sensory_signal function can be called."""

        if not isinstance(regions_name, list):
            raise TypeError(f"unsupported parameter type(s) for regions_name: '{type(regions_name).__name__}'")
            
        self.regions_name = regions_name
        self.region_size = int(region_size)
        self.min_frequency = int(min_frequency)
        self.max_frequency = int(max_frequency)
        self._nb_topographic_regions = len(regions_name)
        self._neurons_by_regions = region_size // self._nb_topographic_regions
        self._timer_ = -1
        self._simulation = None

    def set_simulation(self, simulation: Pong) -> PongSignalTranslator:
        """Setter for the simulation attribute."""
        if not isinstance(simulation, Pong):
            raise TypeError(f"unsupported parameter type(s) for simulation: '{type(simulation).__name__}'")
        self._simulation = simulation
        self._topographic_region_length = simulation.height/self._nb_topographic_regions

        return self
    
    def reset_timer(self) -> None:
        """Resets the timer of the translator object."""
        self._timer_ = -1

    def generate_sensory_signal(self) -> dict[str, list[float]]:
        """Generate the sensory signal from the ball for the Pong simulation."""
        if self._simulation is None:
            raise AttributeError("simulation attribute must be initialize before using this function")
        
        signal_frequency = self.max_frequency + (self._simulation.ball.shape.center.x/float(self._simulation.width))*(self.min_frequency - self.max_frequency)
        signal_period = self._simulation.frequency/signal_frequency

        if self._timer_ == -1 or self._timer_ >= signal_period:
            triggered_region_index = min(self._simulation.ball.shape.center.y // self._topographic_region_length, self._nb_topographic_regions - 1)
            sensory_signal = {region_name: [1.0] * self.region_size if region_index == triggered_region_index else [0.0] * self.region_size for region_index, region_name in enumerate(self.regions_name)}
            self._timer_ = 0
        else:
            sensory_signal = None
            self._timer_ += 1

        return sensory_signal