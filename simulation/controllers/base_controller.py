"""
Base class' module. 

The Controller class defined here should not directly be used as a controller of an element and will make the simulation fail.  
"""

from __future__ import annotations

from simulation.elements.base_element import Element

class Controller:
    """Base class for all Controller objects."""

    def __init__(self):
        """Base class for all Controller objects."""

    def __eq__(self, other) -> bool:
        """Checks if two Controller objects are equal."""
        if isinstance(other, self.__class__):
            self_filtered_dict = {key : value for key, value in self.__dict__.items() if not key.endswith('_')}
            other_filtered_dict = {key : value for key, value in other.__dict__.items() if not key.endswith('_')}
            return self_filtered_dict == other_filtered_dict
        else:
            return False
        
    def __repr__(self) -> str:
        """Controller object's representation."""
        filtered_attributes = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in filtered_attributes.items())})"
    
    def __str__(self) -> str:
        """Controller object's string representation for testing purposes."""        
        return f"{self.__class__.__name__}({self.__dict__})"
        
    def update(self, controlled_element: Element) -> None:
        """Updates controlled element's state."""
        if not isinstance(controlled_element, Element):
            raise TypeError(f"unsupported parameter type(s) for reference: '{type(controlled_element).__name__}'")
