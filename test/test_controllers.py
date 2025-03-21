from unittest import TestCase

from network.exceptions import NetworkCommunicationError as CommunicationError
from network.network import Network
from simulation.controllers.exceptions import ControllerInitializationError as InitializationError
from simulation.controllers.network_controller import ConstantSpeedNetworkController as CSNetworkController
from simulation.controllers.pid_controller import VerticalPositionPIDController as VerticalPID
from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point

class MockNetwork(Network):
    """Simplified Network class for testing purposes."""
    motor_signals: list[tuple[float]] = [(0.0, 0.0), (1.0, 0.3), (0.8, 1.0), (0.2, 0.7)]
    motor_signal_cycle: list[int] = [0, 0, 1, 1, 0, 2, 3, 3]
    signal_cycle_index: int
    regions: tuple[str]

    def __init__(self):
        """Simplified Network class for testing purposes."""
        self.signal_cycle_index = -1
        self.regions = ("foward", "backward", "random")

    def get_motor_signal(self, accessed_regions: tuple[str] | None = None) -> tuple[float]:
        """Simplified get_motor_signal for testing purposes. Iterates through a pre-defined motor signals list each time this function is called."""
        unknown_regions = tuple(set(accessed_regions) - set(self.regions))
        if unknown_regions != ():
            raise CommunicationError("", unknown_regions)
        
        self.signal_cycle_index = (self.signal_cycle_index + 1) % len(self.motor_signal_cycle)

        return self.motor_signals[self.motor_signal_cycle[self.signal_cycle_index]]

class MockElement(Element):
    """Modified element class for testing purposes."""
    def adjust_position(self):
        pass

class TestPIDControllers(TestCase):
    
    def test_vertical_PID_controller(self):
        reference_element_shape = Circle(center=Point(5.0, 0.0), radius=1.0)
        reference_element = MockElement(shape=reference_element_shape, speed=Point(0.0, 1.0))

        controlled_element_shape = Circle(center=Point(0.0, 2.0), radius=1.0)
        controlled_element = MockElement(shape=controlled_element_shape)

        pid_controller = VerticalPID(kp=0.5, ki=1.0, kd=-0.5, reference=reference_element)

        expected_element_shape = Circle(center=Point(0.0, -1.0), radius=1.0)
        expected_element = MockElement(shape=expected_element_shape)

        pid_controller.update(controlled_element)
        self.assertEqual(controlled_element, expected_element)

        reference_element.update()
        
        expected_element_shape = Circle(center=Point(0.0, -2.0), radius=1.0)
        expected_element = MockElement(shape=expected_element_shape)

        pid_controller.update(controlled_element)
        self.assertEqual(controlled_element, expected_element)

class TestNetworkController(TestCase):

    def setUp(self):
        self.network = MockNetwork()
    
    def test_constant_speed_network_controller_initialization(self):
        with self.assertRaises(InitializationError):
            CSNetworkController(network=self.network, accessed_regions=("Unknown", "random"), reference_speed=Point(0.0, 1.0), signal_threshold=0.5)

        with self.assertRaises(InitializationError):
            CSNetworkController(network=self.network, accessed_regions=("foward", "backward", "random"), reference_speed=Point(0.0, 1.0), signal_threshold=0.5)

    def test_constant_speed_network_controller(self):
        controlled_element_shape = Circle(center=Point(0.0, 2.0), radius=1.0)
        controlled_element = MockElement(shape=controlled_element_shape)
        
        network_controller = CSNetworkController(network=self.network, accessed_regions=("foward", "backward"), reference_speed=Point(0.0, 1.0), signal_threshold=0.5)

        possible_positions = [Point(0.0, 2.0), Point(0.0, 3.0), Point(0.0, 4.0)]
        expected_position_cycle = [0, 1, 2, 2, 2, 1, 0]
        
        for i, expected_position in enumerate(expected_position_cycle):
            expected_element_shape = Circle(center=possible_positions[expected_position], radius=1.0)
            expected_element = MockElement(shape=expected_element_shape)

            network_controller.update(controlled_element)

            msg = f"""
            Failed during iteration {i}:
                - expected output: {possible_positions[expected_position]}
                - resulting output: {controlled_element.shape.center}
            """
            
            self.assertEqual(controlled_element, expected_element, msg)
