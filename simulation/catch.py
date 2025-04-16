"""
Catch class module. Inherits from the simulation module.
"""

from __future__ import annotations
from numpy.typing import NDArray

from network.network import Network
from simulation.base_simulation import Simulation
from simulation.elements.ball import Ball
from simulation.elements.paddle import Paddle
from simulation.geometry.point import Point

import json
import numpy as np
import os

from math import tan, radians

class Catch(Simulation):
    """Creates a simulation of the environnement of the game catch, where the ball always follows the same trajectory and the agent must catch it."""
    ball: Ball
    agent: Paddle
    network: Network
    ball_initial_position: Point
    ball_reference_x_speed: float
    ball_reference_speed_orientation: float
    ball_sensory_signal_translator: CatchSignalTranslator
    _agent_initial_position: Point
    _ball_reference_speed: Point
    _success_history_: list[tuple[bool, int]]

    def __init__(
                    self, 
                    height: int,
                    width: int,
                    frequency: int, 
                    ball: Ball,
                    agent: Paddle,
                    network: Network,
                    ball_initial_position: Point,
                    ball_reference_x_speed: float,
                    ball_reference_speed_orientation: float,
                    ball_sensory_signal_translator: CatchSignalTranslator,
                    generator_seed: int,
                    simulation_name: None | str = None
                ):
        """Creates a simulation of the environnement of the arcade game Pong.
            - height: Integer representing the simulation height, in pixels.
            - width: Integer representing the simulation width, in pixels.
            - frequency: Integer representing the abstract frequency of the simulation's iteration process, in Hz.
            - ball: Ball object representing the ball in the simulation.
            - paddle: Paddle object representing the paddle opposing the agent.
            - agent: Paddle object representing the paddle controlled by the agent.
            - network: Network object representing the agent's internal state.
            - ball_initial_position: Point object representing the ball's initial position.
            - ball_reference_speed_norm: Floating value representing the ball speed's norm when regenerated.
            - ball_reference_speed_orientation: Floating value representing the ball speed's orientation when regenerated.
            - ball_generation_area: Shape object representing the area where the ball is regenerated when needed.
            - ball_sensory_signal_translator: CatchSignalTranslator object representing the translator used to communicate with the 
            - generator_seed: Integer representing the seed used when creating the simulation's random number generator."""
        
        if not isinstance(ball, Ball):
            raise TypeError(f"unsupported parameter type(s) for ball: '{type(ball).__name__}'")
        if not isinstance(agent, Paddle):
            raise TypeError(f"unsupported parameter type(s) for agent: '{type(agent).__name__}'")
        if not isinstance(network, Network):
            raise TypeError(f"unsupported parameter type(s) for network: '{type(network).__name__}'")
        if not isinstance(ball_initial_position, Point):
            raise TypeError(f"unsupported parameter type(s) for ball_initial_position: '{type(ball_initial_position).__name__}'")
        if not isinstance(ball_sensory_signal_translator, CatchSignalTranslator):
            raise TypeError(f"unsupported parameter type(s) for ball_sensory_signal_translator: '{type(ball_sensory_signal_translator).__name__}'")
        if not 100.0 < float(ball_reference_speed_orientation) < 260.0:
            raise ValueError(f"The ball reference orientation should be between 100 and 260 degrees, not {ball_reference_x_speed:.0f}.")
        
        self.ball = ball
        self.agent = agent

        super().__init__(height, width, frequency, [self.ball, self.agent], simulation_name=simulation_name, generator_seed=generator_seed)

        self.network = network
        self.ball_initial_position = ball_initial_position
        self.ball_reference_x_speed = float(ball_reference_x_speed)
        self.ball_reference_speed_orientation = float(ball_reference_speed_orientation)
        self.ball_sensory_signal_translator = ball_sensory_signal_translator.set_simulation(self)

        self._agent_initial_position = self.agent.get_position()
        self._ball_reference_speed = Point(-self.ball_reference_x_speed, -self.ball_reference_x_speed * tan(radians(self.ball_reference_speed_orientation)))
        self._success_history_ = []

        self.ball.set_state(position=self.ball_initial_position, speed=self._ball_reference_speed)

    def step(self) -> None:
        super().step()
        self.check_ball_collisions()
        self.network.propagate_signal(self._generator_, self.ball_sensory_signal_translator.generate_sensory_signal())
        self.network.optimize_connections()
        self.network.compute_free_energy()
        
    def check_ball_collisions(self) -> None:
        """Check for ball collisions and resolves its effects, either locally or by calling another method."""
        # Detects collisions with top and bottom walls.
        if (self.ball.shape.center.y <= self.ball.shape.radius) or (self.height - self.ball.shape.center.y <= self.ball.shape.radius):
            reflected_speed = self.ball.speed.reflection(Point(0.0, 1.0))
            self.ball.set_state(speed=reflected_speed)
        # Detects collisions with left wall.
        elif self.ball.shape.center.x <= self.ball.shape.radius:
            self.network.punish(self._generator_)
            self._success_history_.append(np.array([[0.0, self._timer_],]))
            self.reset_agent_position()
            self.regenerate_ball()
            self.ball_sensory_signal_translator.reset_timer()
        # Detects collisions with right wall.
        elif self.width - self.ball.shape.center.x <= self.ball.shape.radius:
            raise ValueError("There should never be any collisions with the right wall.")
        # Detects collisions with the agent.
        elif self.ball.collides_with(self.agent):
            self.resolve_collision_with_agent(self.agent)

    def reset_agent_position(self) -> None:
        """Resets the agent to its initial position."""
        self.agent.set_state(position=self._agent_initial_position)

    def regenerate_ball(self) -> None:
        """Regenerate the ball object at a random position within the simulation ball generation area."""
        self.ball.set_state(position=self.ball_initial_position, speed=self._ball_reference_speed)

    def resolve_collision_with_agent(self, paddle: Paddle) -> None:
        """Resolves the effect of the collision between the ball and the agent."""
        closest_point = paddle.shape.get_closest_point(paddle.shape.translate_to_local(self.ball.shape.center))
        collided_edge_normal_vector = paddle.shape.get_edge_normal_vector(closest_point).rotate(paddle.shape.orientation)

        # Collision with front face
        if collided_edge_normal_vector == Point(1.0, 0.0):
            # Rewards the network and record the agent's success           
            self.network.reward(self._generator_)
            self._success_history_.append(np.array([[1.0, self._timer_],]))
            
            # Regenerates the ball for a new rally
            self.reset_agent_position()
            self.regenerate_ball()
            self.ball_sensory_signal_translator.reset_timer()

        # Collision with other faces
        else:
            speed_adjustment = paddle.speed.projection(collided_edge_normal_vector)
            
            if self.ball.speed * collided_edge_normal_vector <= 0.0:
                ball_speed = self.ball.speed.reflection(collided_edge_normal_vector) + speed_adjustment
            else:
                ball_speed = self.ball.speed + speed_adjustment

            self.ball.set_state(speed=ball_speed)

    def get_success_history(self) -> NDArray:
        return np.concatenate(self._success_history_, axis=0)
    
    def get_average_success_rate(self) -> float:
        """Returns the average success rate of the agent during the simulation."""
        return np.mean(self.get_success_history(), axis=0)[0]
    
    def save_success_history(self, success_history_file_name = "success_history") -> None:
        success_history_file_path = os.path.join(self._simulation_dir_, f"{success_history_file_name}.json")
        with open(success_history_file_path, "w") as success_history_file:
            json.dump(repr(self._success_history_), success_history_file)


class CatchSignalTranslator:
    """This class allows the creation of a sensory signal from a Catch simulation object."""
    regions_name: list[str]
    min_frequency: int
    max_frequency: int
    _nb_topographic_regions: int
    _neurons_by_regions: int
    _topographic_region_length: float
    _timer_: int
    _simulation: Catch

    def __init__(self, regions_name: list[str], region_size: int, min_frequency: int, max_frequency: int):
        """This class allows the creation of a sensory signal from a Catch simulation object.
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

    def set_simulation(self, simulation: Catch) -> CatchSignalTranslator:
        """Setter for the simulation attribute."""
        if not isinstance(simulation, Catch):
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