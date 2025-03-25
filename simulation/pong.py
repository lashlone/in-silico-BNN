"""
Pong class module. Inherits from the simulation module.
"""

from __future__ import annotations

from network.network import Network
from simulation.base_simulation import Simulation
from simulation.elements.ball import Ball
from simulation.elements.paddle import Paddle
from simulation.geometry.point import Point
from simulation.geometry.shape import Shape

class Pong(Simulation):
    """Creates a simulation of the environnement of the arcade game Pong."""
    ball: Ball
    paddle: Paddle
    agent: Paddle
    network: Network
    ball_generation_area: Shape
    ball_sensory_signal_translator: PongSignalTranslator
    ball_min_orientation: float
    ball_max_orientation: float
    _ball_reference_speed: float
    _success_rate_history_: list[tuple[bool, int]]

    def __init__(
                    self, 
                    height: int,
                    width: int,
                    frequency: int, 
                    ball: Ball,
                    paddle: Paddle,
                    agent: Paddle,
                    network: Network,
                    ball_generation_area: Shape,
                    ball_sensory_signal_translator: PongSignalTranslator,
                    generator_seed: int,
                    ball_min_orientation: float = 120.0,
                    ball_max_orientation: float = 240.0
                ):
        """
        Creates a simulation of the environnement of the arcade game Pong.
            - height: size of the simulation's frame on the y-axis, in pixel.
            - width: size of the simulation's frame on the x-axis, in pixel.
            - frequency: the update frequency of the simulation, in Hertz.
            - ball: Ball object, representing the ball in the simulation.
            - paddle: Paddle object, representing the paddle opposing the agent.
            - agent: Paddle object, representing the paddle controlled by the agent.
            - network: Network object, representing the agent's internal state.
            - ball_generation_area: Shape object. The ball in the simulation will always be regenerated in this area when needed.
            - generator_seed: Generator object to use when generating random values.
            - ball_min_orientation (optional): float, representing the ball's orientation minimum when regenerated.
            - ball_max_orientation (optional): float, representing the ball's orientation maximum when regenerated.
        """
        if not isinstance(ball, Ball):
            raise TypeError(f"unsupported parameter type(s) for ball: '{type(ball).__name__}'")
        if not isinstance(paddle, Paddle):
            raise TypeError(f"unsupported parameter type(s) for paddle: '{type(paddle).__name__}'")
        if not isinstance(agent, Paddle):
            raise TypeError(f"unsupported parameter type(s) for agent: '{type(agent).__name__}'")
        if not isinstance(network, Network):
            raise TypeError(f"unsupported parameter type(s) for network: '{type(network).__name__}'")
        if not isinstance(ball_generation_area, Shape):
            raise TypeError(f"unsupported parameter type(s) for ball_generation_area: '{type(ball_generation_area).__name__}'")
        if not isinstance(ball_sensory_signal_translator, PongSignalTranslator):
            raise TypeError(f"unsupported parameter type(s) for ball_sensory_signal_translator: '{type(ball_sensory_signal_translator).__name__}'")
        if not float(ball_min_orientation) < float(ball_max_orientation):
            raise ValueError(f"ball's orientation minimum ({ball_min_orientation}) is bigger or equal to ball's orientation maximum ({ball_max_orientation}).")

        self.ball = ball
        self.paddle = paddle
        self.agent = agent

        super().__init__(height, width, frequency, [self.ball, self.paddle, self.agent], generator_seed=generator_seed)

        self.network = network
        self.ball_generation_area = ball_generation_area
        self.ball_sensory_signal_translator = ball_sensory_signal_translator.set_simulation(self)
        self.ball_min_orientation = ball_min_orientation
        self.ball_max_orientation = ball_max_orientation
        self._ball_reference_speed = self.ball.speed.norm()
        self._success_rate_history_ = []

    def step(self) -> None:
        super().step()
        self.check_ball_collisions()
        self.network.propagate_signal(self._generator_, self.ball_sensory_signal_translator.generate_sensory_signal())
        self.network.optimize_connections()
        self.network.compute_free_energy()
        
    def check_ball_collisions(self) -> None:
        """Check for ball collisions and resolves its effects, either locally or by calling another method."""
        # Detects collision with top and bottom walls
        if (self.ball.shape.center.y <= self.ball.shape.radius) or (self.height - self.ball.shape.center.y <= self.ball.shape.radius):
            reflected_speed = self.ball.speed.reflection(Point(0.0, 1.0))
            self.ball.set_state(speed=reflected_speed)
        elif self.ball.shape.center.x <= self.ball.shape.radius:
            self.network.punish(self._generator_)
            self._success_rate_history_.append((False, self._timer_))
            self.regenerate_ball()
            self.ball_sensory_signal_translator.reset_timer()
        elif self.width - self.ball.shape.center.x <= self.ball.shape.radius:
            self.network.reward()
            self.regenerate_ball()
            self.ball_sensory_signal_translator.reset_timer()
        elif self.ball.collides_with(self.paddle):
            self.resolve_collision_with_paddle(self.paddle)
        elif self.ball.collides_with(self.agent):
            self.network.reward()
            self._success_rate_history_.append((True, self._timer_))
            self.resolve_collision_with_paddle(self.agent)

    def regenerate_ball(self) -> None:
        """Regenerate the ball object at a random position within the simulation ball generation area."""
        ball_position = self.ball_generation_area.get_random_point(self._generator_)
        ball_speed_orientation = self._generator_.uniform(low=self.ball_min_orientation, high=self.ball_max_orientation)
        ball_speed = Point(self._ball_reference_speed, 0.0).rotate(ball_speed_orientation)
        self.ball.set_state(position=ball_position, speed=ball_speed)

    def resolve_collision_with_paddle(self, paddle: Paddle):
        """Resolves the effect of the collision between the ball and a paddle object"""
        closest_point = paddle.shape.get_closest_point(paddle.shape.translate_to_local(self.ball.shape.center))
        collided_edge_normal_vector = paddle.shape.get_edge_normal_vector(closest_point).rotate(paddle.shape.orientation)

        speed_adjustment = paddle.speed.projection(collided_edge_normal_vector)

        if self.ball.speed * collided_edge_normal_vector <= 0.0:
            ball_speed = self.ball.speed.reflection(collided_edge_normal_vector) + speed_adjustment
        else:
            ball_speed = self.ball.speed + speed_adjustment

        self.ball.set_state(speed=ball_speed)


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
        """
        This class allows the creation of a sensory signal from a Pong simulation object.
            - region_name: name of the associated signal region in the network.
            - region_size: size of the associated signal region in the network.
            - min_frequency: minimum frequency of the signal, when the ball is farthest from the agent.
            - max_frequency: maximum frequency of the signal, when the ball is closest to the agent.
            - nb_topographic_regions: number of topographic regions in the associated signal region. Defines how to split the vertical grid into regions.

        This class's simulation attribute has to be defined through its setter before the generate_sensory_signal function can be called. 
        """
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