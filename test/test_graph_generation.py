from unittest import TestCase
from numpy.testing import assert_equal

from network.graph_generation import fixed_average_transmission, self_referring_fixed_average_transmission

import numpy as np
from numpy.random import Generator

class TestGenerator(Generator):
    """Non-stochastic generator for testing purposes."""
    fixed_value: float

    def __init__(self, fixed_value: float):
        """
        Non-stochastic generator for testing purposes.
            - fixed_value : value the generator should always return.
        """
        self.fixed_value = float(fixed_value)

    def uniform(self, low=None, high=None, size=None):
        if size is None:
            return self.fixed_value
        else:
            return self.fixed_value * np.ones(size)

class TestBaseElement(TestCase):
    def setUp(self):
        self.generator = TestGenerator(0.5)

    def test_fixed_average_transmission(self):
        with self.assertRaises(ValueError):
            fixed_average_transmission(1.25, self.generator)
        
        expected_connections = np.array([[0.66, 0.66], [0.66, 0.66], [0.66, 0.66],]).astype(np.float32)

        result_fn = fixed_average_transmission(0.66, self.generator)
        result_connections = result_fn(3, 2)

        assert_equal(result_connections, expected_connections)

    def test_self_referring_fixed_average_transmission(self):
        with self.assertRaises(ValueError):
            self_referring_fixed_average_transmission(-0.25, self.generator)

        expected_connections = np.array([[np.nan, 0.4], [0.4, np.nan]]).astype(np.float32)

        result_fn = self_referring_fixed_average_transmission(0.4, self.generator)

        with self.assertRaises(ValueError):
            result_fn(10, 1)
        result_connections = result_fn(2, 2)

        assert_equal(result_connections, expected_connections)