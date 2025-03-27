from network.graph_generation import fixed_average_transmission, self_referring_fixed_average_transmission
from network.regions import ExternalRegion, InternalRegion
from network.network import Network

from simulation.controllers.network_controller import ConstantSpeedNetworkController as NetworkController
from simulation.controllers.pid_controller import VerticalPositionPIDController as PIDController
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

SENSORY_REGION_SIZE = 1
AFFERENT_REGION_SIZE = 4
INTERNAL_REGION_SIZE = 64
EFFERENT_REGION_SIZE = 12

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

# Simulation's default values
HEIGHT = 300
WIDTH = 400
FREQUENCY = 240

BALL_RADIUS = 7.5
PADDLE_CONTROLLER_KP = 1.0
PADDLE_CONTROLLER_KI = 0.0
PADDLE_CONTROLLER_KD = 0.0
PAD_X = 20.0
PAD_Y = 20.0
AGENT_SPEED = Point(0.0, 3.0)
AGENT_CONTROLLER_THRESHOLD = 0.33
SENSORY_SIGNAL_MIN_FREQUENCY = 30
SENSORY_SIGNAL_MAX_FREQUENCY = 60
SIMULATION_GENERATOR_SPEED = 7777

def init_simulation(decay_coefficient: float, exploration_rate: float, strengthening_rate: float, simulation_name: str):
    sensory_region_names = [f's{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    afferent_region_names = [f'a{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    internal_region_names = ['i0',]
    efferent_region_names = ['e0', 'e1',]

    sensory_regions = [ExternalRegion(name=region_name, size=SENSORY_REGION_SIZE) for region_name in sensory_region_names]
    afferent_regions = [InternalRegion(name=region_name, size=AFFERENT_REGION_SIZE) for region_name in afferent_region_names]
    efferent_regions = [InternalRegion(name=region_name, size=EFFERENT_REGION_SIZE) for region_name in efferent_region_names]
    internal_regions = [InternalRegion(name=region_name, size=INTERNAL_REGION_SIZE) for region_name in internal_region_names]   
    
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

    agent_area_center = Point(WIDTH/2.0, HEIGHT/2.0)
    
    ball_speed = Point(-2.0, 2.0)
    ball_acceleration = Point(0.0, 0.0)
    ball_speed_range = (0.00003, 12.0)

    ball = Ball(shape=Circle(center=agent_area_center, radius=BALL_RADIUS), speed=ball_speed, speed_range=ball_speed_range, acceleration=ball_acceleration)

    paddle_width = 15.0
    paddle_height = 72.0
    paddle_shape_center = Point(WIDTH - (PAD_X + paddle_width/2.0), HEIGHT/2.0)
    paddle_shape = Rectangle(center=paddle_shape_center, width=paddle_width, height=paddle_height, orientation=180.0)
    paddle_controller = PIDController(kp=PADDLE_CONTROLLER_KP, ki=PADDLE_CONTROLLER_KI, kd=PADDLE_CONTROLLER_KD, reference=ball)
    paddle_y_range = (PAD_Y + paddle_height/2.0, HEIGHT - (PAD_Y + paddle_height/2.0))

    paddle = Paddle(shape=paddle_shape, controller=paddle_controller, y_range=paddle_y_range)

    agent_shape_center = Point(PAD_X + paddle_width/2.0, HEIGHT/2.0)
    agent_shape = Rectangle(center=agent_shape_center, width=paddle_width, height=paddle_height, orientation=0.0)
    agent_controller = NetworkController(network=network, accessed_regions=tuple(efferent_region_names), reference_speed=AGENT_SPEED, signal_threshold=AGENT_CONTROLLER_THRESHOLD)
    agent = Paddle(shape=agent_shape, controller=agent_controller, y_range=paddle_y_range)

    ball_generation_area = Rectangle(center=agent_area_center, width=WIDTH/4.0, height=3.0*HEIGHT/4.0)

    ball_sensory_signal_translator = PongSignalTranslator(sensory_region_names, SENSORY_REGION_SIZE, SENSORY_SIGNAL_MIN_FREQUENCY, SENSORY_SIGNAL_MAX_FREQUENCY)

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