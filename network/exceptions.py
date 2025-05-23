"""
Network exception classes' module.
"""

class NetworkInitializationError(Exception):
    """Includes any error encountered during the network's initialization."""

class NetworkCommunicationError(Exception):
    """Includes any error when accessed regions' name and/or sizes did not match expected value."""

    def __init__(self, msg: str, faulty_regions: tuple[str], *args):
        """Includes any error when accessed regions' name and/or sizes did not match expected value. The faulty regions should be given as a parameter."""
        super().__init__(msg, *args)
        self.faulty_regions = faulty_regions