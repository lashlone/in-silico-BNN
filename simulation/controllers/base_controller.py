"""
Base class module. 

The Controller class defined here should not directly be used as a controller of an element and will make the simulation fail.  
"""

from simulation.elements.base_element import Element

class Controller:
    """Base class for all Controller objects."""

    def __init__(self):
        """Base class for all Controller objects."""
        
    def update(self, controlled_element: Element):
        """Updates controlled element's state."""
        if not isinstance(controlled_element, Element):
            raise TypeError(f"unsupported parameter type(s) for reference: '{type(controlled_element).__name__}'")
