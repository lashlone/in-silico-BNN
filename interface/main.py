from network.graph_generation import fixed_average_transmission, self_referring_fixed_average_transmission
from network.region import Region
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
STATE_HISTORY_SIZE = 12
DECAY_RATE = 0.005
EXPLORATION_RATE = 0.02
STRENGTHENING_RATE = 0.33

# Simulation's default values
HEIGHT = 800
WIDTH = 1440
FREQUENCY = 240

ball_radius = 10.0
paddle_controller_kp = 1.0
paddle_controller_ki = 0.0
paddle_controller_kd = 0.0
agent_speed = Point(0.0, 0.2)
agent_controller_threshold = 0.0833
sensory_signal_min_frequency = 10
sensory_signal_max_frequency = 60
simulation_generator_seed = 777

def main():
    sensory_region_names = [f's{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    afferent_region_names = [f'a{i}' for i in range(NB_TOPOGRAPHIC_REGIONS)]
    internal_region_names = ['i0',]
    efferent_region_names = ['e0', 'e1',]

    network_external_regions_names = sensory_region_names
    network_internal_regions_names = afferent_region_names + internal_region_names + efferent_region_names
    network_regions_names = network_external_regions_names + network_internal_regions_names

    sensory_region_size = 1
    afferent_region_size = 4
    internal_region_size = 64
    efferent_region_size = 12

    sensory_regions = [Region(name=region_name, size=sensory_region_size, internal=False) for region_name in sensory_region_names]
    afferent_regions = [Region(name=region_name, size=afferent_region_size, internal=True) for region_name in afferent_region_names]
    efferent_regions = [Region(name=region_name, size=efferent_region_size, internal=True) for region_name in efferent_region_names]
    internal_regions = [Region(name=region_name, size=internal_region_size, internal=True) for region_name in internal_region_names]   
    
    regions = sensory_regions + afferent_regions + internal_regions + efferent_regions

    connectome_generator = np.random.default_rng()

    sensory_to_afferent_transmission_average = 0.7
    afferent_to_afferent_transmission_average = 0.1
    afferent_to_efferent_transmission_average = 0.05
    afferent_to_internal_transmission_average = 0.5
    afferent_to_self_transmission_average = 0.2
    efferent_to_afferent_transmission_average = 0.05
    efferent_to_efferent_transmission_average = 0.2
    efferent_to_internal_transmission_average = 0.5
    efferent_to_self_transmission_average = 0.1
    internal_to_afferent_transmission_average = 0.4
    internal_to_efferent_transmission_average = 0.4
    internal_to_self_transmission_average = 0.5

    sensory_to_afferent = fixed_average_transmission(sensory_to_afferent_transmission_average, connectome_generator)
    afferent_to_afferent = fixed_average_transmission(afferent_to_afferent_transmission_average, connectome_generator)
    afferent_to_efferent = fixed_average_transmission(afferent_to_efferent_transmission_average, connectome_generator)
    afferent_to_internal = fixed_average_transmission(afferent_to_internal_transmission_average, connectome_generator)
    afferent_to_self = self_referring_fixed_average_transmission(afferent_to_self_transmission_average, connectome_generator)
    efferent_to_afferent = fixed_average_transmission(efferent_to_afferent_transmission_average, connectome_generator)
    efferent_to_efferent = fixed_average_transmission(efferent_to_efferent_transmission_average, connectome_generator)
    efferent_to_internal = fixed_average_transmission(efferent_to_internal_transmission_average, connectome_generator)
    efferent_to_self = self_referring_fixed_average_transmission(efferent_to_self_transmission_average, connectome_generator)
    internal_to_afferent = fixed_average_transmission(internal_to_afferent_transmission_average, connectome_generator)
    internal_to_efferent = fixed_average_transmission(internal_to_efferent_transmission_average, connectome_generator)
    internal_to_internal = None
    internal_to_self = self_referring_fixed_average_transmission(internal_to_self_transmission_average, connectome_generator)

    afferent_graph_generation_fns = [(afferent_region_names, afferent_to_afferent), (internal_region_names, afferent_to_internal), (efferent_region_names, afferent_to_efferent)]
    efferent_graph_generation_fns = [(afferent_region_names, efferent_to_afferent), (internal_region_names, efferent_to_internal), (efferent_region_names, efferent_to_efferent)]
    internal_graph_generation_fns = [(afferent_region_names, internal_to_afferent), (internal_region_names, internal_to_internal), (efferent_region_names, internal_to_efferent)]
    
    afferent_regions_connections = {name: {region_name: fn if region_name != name else afferent_to_self for region_names, fn in afferent_graph_generation_fns for region_name in region_names} for name in afferent_region_names}
    efferent_regions_connections = {name: {region_name: fn if region_name != name else efferent_to_self for region_names, fn in efferent_graph_generation_fns for region_name in region_names} for name in efferent_region_names}
    internal_regions_connections = {name: {region_name: fn if region_name != name else internal_to_self for region_names, fn in internal_graph_generation_fns for region_name in region_names} for name in internal_region_names}
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
    
    regions_connectome = {**sensory_regions_connections, **afferent_regions_connections, **efferent_regions_connections, **internal_regions_connections}

    network = Network(
        regions=regions, 
        regions_connectome=regions_connectome,
        recovery_state_energy_ratio=RECOVERY_STATE_ENERGY_RATIO,
        state_history_size=STATE_HISTORY_SIZE,
        decay_rate=DECAY_RATE,
        exploration_rate=EXPLORATION_RATE,
        strengthening_rate=STRENGTHENING_RATE,
    )

    agent_area_center = Point(360.0, 400.0)
    pad_x = 30.0
    pad_y = 30.0

    ball_acceleration = Point(0.0, 0.0)
    ball = Ball(shape=Circle(center=agent_area_center, radius=ball_radius), speed=Point(0.33, 0.33), acceleration=ball_acceleration, speed_range=(0.000005, 0.5))

    paddle_width = 20.0
    paddle_height = 120.0
    paddle_shape_center = Point(1440.0 - (pad_x + paddle_width/2.0), 400.0)
    paddle_y_range = (pad_y + paddle_height/2.0, 1440.0 - (pad_y + paddle_height/2.0))
    paddle_shape = Rectangle(center=paddle_shape_center, width=paddle_width, height=paddle_height, orientation=180.0)
    paddle = Paddle(shape=paddle_shape, controller=PIDController(kp=paddle_controller_kp, ki=paddle_controller_ki, kd=paddle_controller_kd, reference=ball), y_range=paddle_y_range)

    agent_shape_center = Point(pad_x + paddle_width/2.0, 400.0)
    agent_shape = Rectangle(center=agent_shape_center, width=paddle_width, height=paddle_height, orientation=0.0)
    agent = Paddle(shape=agent_shape, controller=NetworkController(network=network, accessed_regions=tuple(efferent_region_names), reference_speed=agent_speed, signal_threshold=agent_controller_threshold), y_range=paddle_y_range)

    ball_generation_area = Rectangle(center=agent_area_center, width=240.0, height=600.0)

    ball_sensory_signal_translator = PongSignalTranslator(sensory_region_names, sensory_region_size, sensory_signal_min_frequency, sensory_signal_max_frequency)

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
        generator_seed=simulation_generator_seed
    )

    return simulation