"""
This module defines functions to initialize the network, the catch simulation and the pong simulation.
"""

from network.graph_generation import fixed_average_transmission, self_referring_fixed_average_transmission
from network.regions import ExternalRegion, InternalRegion
from network.network import Network

from simulation.catch import Catch, CatchSignalTranslator
from simulation.controllers.network_controller import ConstantSpeedNetworkController as NetworkController
from simulation.controllers.pid_controller import VerticalPositionPIDController as PIDController
from simulation.controllers.random_controller import LinearRandomWalker as RWController
from simulation.elements.ball import Ball
from simulation.elements.paddle import Paddle
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point
from simulation.geometry.rectangle import Rectangle
from simulation.pong import Pong, PongSignalTranslator

import numpy as np

NB_TOPOGRAPHIC_REGIONS = 8

# Networks' default values
RECOVERY_STATE_ENERGY_RATIO = 0.5
STATE_HISTORY_SIZE = 10
DECAY_COEFFICIENT = 0.02
EXPLORATION_RATE = 0.01
STRENGTHENING_RATE = 1.1

SENSORY_TO_AFFERENT_TRANSMISSION_AVERAGE = 0.75
AFFERENT_TO_AFFERENT_TRANSMISSION_AVERAGE = 0.05
AFFERENT_TO_EFFERENT_TRANSMISSION_AVERAGE = 0.025
AFFERENT_TO_INTERNAL_TRANSMISSION_AVERAGE = 0.4
AFFERENT_TO_SELF_TRANSMISSION_AVERAGE = 0.1
EFFERENT_TO_AFFERENT_TRANSMISSION_AVERAGE = 0.025
EFFERENT_TO_EFFERENT_TRANSMISSION_AVERAGE = 0.1
EFFERENT_TO_INTERNAL_TRANSMISSION_AVERAGE = 0.4
EFFERENT_TO_SELF_TRANSMISSION_AVERAGE = 0.05
INTERNAL_TO_AFFERENT_TRANSMISSION_AVERAGE = 0.35
INTERNAL_TO_EFFERENT_TRANSMISSION_AVERAGE = 0.35
INTERNAL_TO_SELF_TRANSMISSION_AVERAGE = 0.5

# Simulations' default values
HEIGHT = 320
WIDTH = 400
FREQUENCY = 240

PAD_X = 20.0
PAD_Y = 10.0
BALL_RADIUS = 7.5
BALL_SPEED_RANGE = (0.0005, 12.0)
AGENT_SPEED = Point(0.0, 3.0)
AGENT_CONTROLLER_THRESHOLD = 0.33
SENSORY_SIGNAL_MIN_FREQUENCY = 30
SENSORY_SIGNAL_MAX_FREQUENCY = 60
SIMULATION_GENERATOR_SPEED = 7777

# Pong simulation's default values
PONG_PADDLE_CONTROLLER_KP = 1.0
PONG_PADDLE_CONTROLLER_KI = 0.0
PONG_PADDLE_CONTROLLER_KD = 0.0

# Catch simulation's default values
BALL_INITIAL_POSITION = Point(300.0, 160.0)
BALL_X_SPEED = 2.0

def init_network(decay_coefficient: float, exploration_rate: float, strengthening_rate: float, regions_size: dict[str, int]) -> tuple[list[str], list[str], Network]:
    sensory_region_names = [f's{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    afferent_region_names = [f'a{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    internal_region_names = ['i0',]
    efferent_region_names = ['e0', 'e1',]

    sensory_regions = [ExternalRegion(name=region_name, size=regions_size['sensory']) for region_name in sensory_region_names]
    afferent_regions = [InternalRegion(name=region_name, size=regions_size['afferent']) for region_name in afferent_region_names]
    efferent_regions = [InternalRegion(name=region_name, size=regions_size['efferent']) for region_name in efferent_region_names]
    internal_regions = [InternalRegion(name=region_name, size=regions_size['internal']) for region_name in internal_region_names]
    
    regions = sensory_regions + afferent_regions + internal_regions + efferent_regions

    connectome_generator = np.random.default_rng()

    sensory_to_afferent = fixed_average_transmission(SENSORY_TO_AFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    afferent_to_afferent = fixed_average_transmission(AFFERENT_TO_AFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    afferent_to_efferent = fixed_average_transmission(AFFERENT_TO_EFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    afferent_to_internal = fixed_average_transmission(AFFERENT_TO_INTERNAL_TRANSMISSION_AVERAGE, connectome_generator)
    afferent_to_self = self_referring_fixed_average_transmission(AFFERENT_TO_SELF_TRANSMISSION_AVERAGE, connectome_generator)
    efferent_to_afferent = fixed_average_transmission(EFFERENT_TO_AFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    efferent_to_efferent = fixed_average_transmission(EFFERENT_TO_EFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    efferent_to_internal = fixed_average_transmission(EFFERENT_TO_INTERNAL_TRANSMISSION_AVERAGE, connectome_generator)
    efferent_to_self = self_referring_fixed_average_transmission(EFFERENT_TO_SELF_TRANSMISSION_AVERAGE, connectome_generator)
    internal_to_afferent = fixed_average_transmission(INTERNAL_TO_AFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    internal_to_efferent = fixed_average_transmission(INTERNAL_TO_EFFERENT_TRANSMISSION_AVERAGE, connectome_generator)
    internal_to_internal = None
    internal_to_self = self_referring_fixed_average_transmission(INTERNAL_TO_SELF_TRANSMISSION_AVERAGE, connectome_generator)

    afferent_graph_generation_fns = [(afferent_region_names, afferent_to_afferent), (internal_region_names, afferent_to_internal), (efferent_region_names, afferent_to_efferent)]
    efferent_graph_generation_fns = [(afferent_region_names, efferent_to_afferent), (internal_region_names, efferent_to_internal), (efferent_region_names, efferent_to_efferent)]
    internal_graph_generation_fns = [(afferent_region_names, internal_to_afferent), (internal_region_names, internal_to_internal), (efferent_region_names, internal_to_efferent)]
    
    sensory_regions_connections = {
        's0': {'a0': sensory_to_afferent},
        's1': {'a1': sensory_to_afferent},
        's2': {'a2': sensory_to_afferent},
        's3': {'a3': sensory_to_afferent},
        's4': {'a4': sensory_to_afferent},
        's5': {'a5': sensory_to_afferent},
        's6': {'a6': sensory_to_afferent},
        's7': {'a7': sensory_to_afferent},
    }

    afferent_regions_connections = {name: {region_name: fn if region_name != name else afferent_to_self for region_names, fn in afferent_graph_generation_fns for region_name in region_names} for name in afferent_region_names}
    efferent_regions_connections = {name: {region_name: fn if region_name != name else efferent_to_self for region_names, fn in efferent_graph_generation_fns for region_name in region_names} for name in efferent_region_names}
    internal_regions_connections = {name: {region_name: fn if region_name != name else internal_to_self for region_names, fn in internal_graph_generation_fns for region_name in region_names} for name in internal_region_names}
    
    regions_connectome = {**sensory_regions_connections, **afferent_regions_connections, **efferent_regions_connections, **internal_regions_connections}

    network = Network(
        regions=regions, 
        regions_connectome=regions_connectome,
        recovery_state_energy_ratio=RECOVERY_STATE_ENERGY_RATIO,
        state_history_size=STATE_HISTORY_SIZE,
        decay_coefficient=decay_coefficient,
        exploration_rate=exploration_rate,
        strengthening_exponent=strengthening_rate,
    )
    
    return sensory_region_names, efferent_region_names, network

def init_PID_pong_simulation(decay_coefficient: float, exploration_rate: float, strengthening_rate: float, agent_controller_threshold: float, simulation_name: str, regions_size: dict[str, int]):
    sensory_region_names, efferent_region_names, network = init_network(decay_coefficient, exploration_rate, strengthening_rate, regions_size)

    ball_area_center = Point(WIDTH/2.0, HEIGHT/2.0)
    
    ball_speed = Point(-2.0, 2.0)
    ball_acceleration = Point(0.0, 0.0)

    ball = Ball(shape=Circle(center=ball_area_center, radius=BALL_RADIUS), speed=ball_speed, speed_range=BALL_SPEED_RANGE, acceleration=ball_acceleration)

    paddle_width = 15.0
    paddle_height = 60.0
    paddle_shape_center = Point(WIDTH - (PAD_X + paddle_width/2.0), HEIGHT/2.0)
    paddle_shape = Rectangle(center=paddle_shape_center, width=paddle_width, height=paddle_height, orientation=180.0)
    paddle_controller = PIDController(kp=PONG_PADDLE_CONTROLLER_KP, ki=PONG_PADDLE_CONTROLLER_KI, kd=PONG_PADDLE_CONTROLLER_KD, reference=ball)
    paddle_y_range = (PAD_Y + paddle_height/2.0, HEIGHT - (PAD_Y + paddle_height/2.0))

    paddle = Paddle(shape=paddle_shape, controller=paddle_controller, y_range=paddle_y_range)

    agent_shape_center = Point(PAD_X + paddle_width/2.0, HEIGHT/2.0)
    agent_shape = Rectangle(center=agent_shape_center, width=paddle_width, height=paddle_height, orientation=0.0)
    agent_controller = NetworkController(network=network, accessed_regions=tuple(efferent_region_names), reference_speed=AGENT_SPEED, signal_threshold=agent_controller_threshold)
    agent = Paddle(shape=agent_shape, controller=agent_controller, y_range=paddle_y_range)

    ball_generation_area = Rectangle(center=ball_area_center, width=WIDTH/4.0, height=3.0*HEIGHT/4.0)

    ball_sensory_signal_translator = PongSignalTranslator(sensory_region_names, regions_size['sensory'], SENSORY_SIGNAL_MIN_FREQUENCY, SENSORY_SIGNAL_MAX_FREQUENCY)

    simulation = Pong(
        height=HEIGHT,
        width=WIDTH,
        frequency=FREQUENCY,
        ball=ball,
        paddle=paddle,
        agent=agent,
        network=network,
        ball_generation_area=ball_generation_area,
        ball_sensory_signal_translator=ball_sensory_signal_translator,
        generator_seed=SIMULATION_GENERATOR_SPEED,
        simulation_name=simulation_name
    )

    return simulation

def init_random_pong_simulation(decay_coefficient: float, exploration_rate: float, strengthening_rate: float, agent_controller_threshold: float, simulation_name: str, regions_size: dict[str, int]):
    sensory_region_names, efferent_region_names, network = init_network(decay_coefficient, exploration_rate, strengthening_rate, regions_size)

    ball_area_center = Point(WIDTH/2.0, HEIGHT/2.0)
    
    ball_speed = Point(-2.0, 2.0)
    ball_acceleration = Point(0.0, 0.0)

    ball = Ball(shape=Circle(center=ball_area_center, radius=BALL_RADIUS), speed=ball_speed, speed_range=BALL_SPEED_RANGE, acceleration=ball_acceleration)

    paddle_width = 15.0
    paddle_height = 60.0
    paddle_shape_center = Point(WIDTH - (PAD_X + paddle_width/2.0), HEIGHT/2.0)
    paddle_shape = Rectangle(center=paddle_shape_center, width=paddle_width, height=paddle_height, orientation=180.0)
    paddle_controller = RWController(reference_speed=AGENT_SPEED)
    paddle_y_range = (PAD_Y + paddle_height/2.0, HEIGHT - (PAD_Y + paddle_height/2.0))

    paddle = Paddle(shape=paddle_shape, controller=paddle_controller, y_range=paddle_y_range)

    agent_shape_center = Point(PAD_X + paddle_width/2.0, HEIGHT/2.0)
    agent_shape = Rectangle(center=agent_shape_center, width=paddle_width, height=paddle_height, orientation=0.0)
    agent_controller = NetworkController(network=network, accessed_regions=tuple(efferent_region_names), reference_speed=AGENT_SPEED, signal_threshold=agent_controller_threshold)
    agent = Paddle(shape=agent_shape, controller=agent_controller, y_range=paddle_y_range)

    ball_generation_area = Rectangle(center=ball_area_center, width=WIDTH/4.0, height=3.0*HEIGHT/4.0)

    ball_sensory_signal_translator = PongSignalTranslator(sensory_region_names, regions_size['sensory'], SENSORY_SIGNAL_MIN_FREQUENCY, SENSORY_SIGNAL_MAX_FREQUENCY)

    simulation = Pong(
        height=HEIGHT,
        width=WIDTH,
        frequency=FREQUENCY,
        ball=ball,
        paddle=paddle,
        agent=agent,
        network=network,
        ball_generation_area=ball_generation_area,
        ball_sensory_signal_translator=ball_sensory_signal_translator,
        generator_seed=SIMULATION_GENERATOR_SPEED,
        simulation_name=simulation_name
    )

    simulation.paddle.controller.set_generator(simulation._generator_)
    
    return simulation

def init_catch_simulation(ball_speed_orientation: float, decay_coefficient: float, exploration_rate: float, strengthening_rate: float, agent_controller_threshold: float, simulation_name: str, regions_size: dict[str, int]):
    sensory_region_names, efferent_region_names, network = init_network(decay_coefficient, exploration_rate, strengthening_rate, regions_size)
    
    ball_speed = Point(0.0, 0.0)
    ball_acceleration = Point(0.0, 0.0)

    ball = Ball(shape=Circle(center=BALL_INITIAL_POSITION, radius=BALL_RADIUS), speed=ball_speed, speed_range=BALL_SPEED_RANGE, acceleration=ball_acceleration)

    paddle_width = 15.0
    paddle_height = 60.0
    paddle_y_range = (PAD_Y + paddle_height/2.0, HEIGHT - (PAD_Y + paddle_height/2.0))

    agent_shape_center = Point(PAD_X + paddle_width/2.0, HEIGHT/2.0)
    agent_shape = Rectangle(center=agent_shape_center, width=paddle_width, height=paddle_height, orientation=0.0)
    agent_controller = NetworkController(network=network, accessed_regions=tuple(efferent_region_names), reference_speed=AGENT_SPEED, signal_threshold=agent_controller_threshold)
    agent = Paddle(shape=agent_shape, controller=agent_controller, y_range=paddle_y_range)

    ball_sensory_signal_translator = CatchSignalTranslator(sensory_region_names, regions_size['sensory'], SENSORY_SIGNAL_MIN_FREQUENCY, SENSORY_SIGNAL_MAX_FREQUENCY)

    simulation = Catch(
        height=HEIGHT,
        width=WIDTH,
        frequency=FREQUENCY,
        ball=ball,
        agent=agent,
        network=network,
        ball_initial_position=BALL_INITIAL_POSITION,
        ball_reference_x_speed=BALL_X_SPEED,
        ball_reference_speed_orientation=ball_speed_orientation,
        ball_sensory_signal_translator=ball_sensory_signal_translator,
        generator_seed=SIMULATION_GENERATOR_SPEED,
        simulation_name=simulation_name
    )

    return simulation