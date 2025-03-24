"""
NetworkController class module. Inherits from the Controller class. Used to transform the motor signal from the network into an action on the controlled element.
"""

from network.exceptions import NetworkCommunicationError as CommunicationError
from network.network import Network
from simulation.controllers.base_controller import Controller
from simulation.controllers.exceptions import ControllerInitializationError as InitializationError
from simulation.elements.base_element import Element
from simulation.geometry.point import Point

class NetworkController(Controller):
    """Base class for NetworkController objects."""
    network: Network
    accessed_regions: tuple[str]

    def __init__(self, network: Network, accessed_regions: tuple[str]):
        """Base class for NetworkController objects.
            - network: Network object representing the referenced network.
            - accessed_regions : Tuple of strings representing the accessed motor regions from the network."""
        
        if not isinstance(network, Network):
            raise TypeError(f"unsupported parameter type(s) for network: '{type(network).__name__}'")
        if not isinstance(accessed_regions, tuple):
            raise TypeError(f"unsupported parameter type(s) for accessed_regions: '{type(accessed_regions).__name__}'")
        try:
            network.get_motor_signal(accessed_regions)
        except CommunicationError as error:
            raise InitializationError(f"The accessed region(s) {list(error.faulty_regions)} does not exist in the given network.")
        
        self.network = network
        self.accessed_regions = accessed_regions

class ConstantSpeedNetworkController(NetworkController):
    """Moves the element vertically by a fixed speed based on the average firing in the motor regions."""
    reference_speed: Point
    signal_threshold: float

    def __init__(self, network: Network, accessed_regions: tuple[str], reference_speed: Point, signal_threshold: float):
        """NetworkController that moves the element vertically by a fixed speed based on the average firing in the motor region.
            - network: Network object representing the referenced network.
            - accessed_regions : Tuple of strings representing the two accessed motor regions from the network. The first region is for forward motion while the second is for backward motion.
            - reference_speed : Point object representing the unitary forward or backward motion.
            - signal_threshold : Floating value representing the threshold that the signal must exceed in order to make the controlled element move."""
        
        super().__init__(network, accessed_regions)
        
        if not len(accessed_regions) == 2:
            raise InitializationError(f"Expected to access 2 motor regions and got {len(accessed_regions)} instead.")
        if not isinstance(reference_speed, Point):
            raise TypeError(f"unsupported parameter type(s) for reference_speed: '{type(reference_speed).__name__}'")
        
        self.reference_speed = reference_speed
        self.signal_threshold = float(signal_threshold)

    def update(self, controlled_element: Element) -> None:
        super().update(controlled_element)

        # Gets the signal for moving foward or moving backward
        forward_signal, backward_signal = self.network.get_motor_signal(self.accessed_regions)

        # Moves the element based on the given threshold
        if forward_signal >= self.signal_threshold:
            controlled_element.shape.move_center(self.reference_speed)
        if backward_signal >= self.signal_threshold:
            controlled_element.shape.move_center(-self.reference_speed)