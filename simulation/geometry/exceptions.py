"""
Exception classes module.
"""

class EdgeError(Exception):
    """Includes any error encountered while using shape's edges."""

class CurvedEdgeError(Exception):
    """Corners are not defined on curved lines."""