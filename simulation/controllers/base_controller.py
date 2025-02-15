"""
Base class module. 

The Controller class defined here should not directly be used as a controller of an element and will make the simulation fail.  
"""

class Controller:
    """Base class for all Controller objects."""

    def __init__(self):
        """Base class for all Controller objects."""
        
    def update(self, *args):
        """Updates controlled element's state."""
        raise NotImplementedError("Subclasses must implement this method.")
