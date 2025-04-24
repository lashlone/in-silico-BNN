import unittest
from numpy.testing import assert_equal

from network.exceptions import NetworkCommunicationError
from network.network import Network
from network.regions import InternalRegion, ExternalRegion

import numpy as np
from numpy.random import Generator   

test_network_state = np.array([1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0])
test_network_conformation = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                      [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                      [np.nan, np.nan, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
                                      [np.nan, np.nan, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0],
                                      [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
                                      [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],])

class MockGenerator(Generator):
    """Non-stochastic generator for testing purposes."""

    def __init__(self):    
        """Non-stochastic generator for testing purposes."""

    def choice(self, array, choices, replace):
        return [array - (i+1) for i in range(choices)]
    
    def integers(self, low, high, size=None):
        return np.full(size, (low + high) // 2)
    
    def uniform(self):
        return 0.4


class TestNetwork(unittest.TestCase):
    def setUp(self):
        regions = [ExternalRegion("region1", 2), InternalRegion("region2", 7)]
        regions_connectome = {"region1": {"region2": lambda x, y: np.ones((x, y)) * 0.2}, "region2": {"region2": lambda x, y: np.ones((x, y)) * 0.2}}
        self.network = Network(regions, regions_connectome, k_value=1, state_history_size=2, decay_coefficient=0.02, exploration_rate=0.01, strengthening_exponent=1.1)
        self.network.set_state(test_network_state)
        self.network._conformation = self.network._conformation * test_network_conformation
        self.generator = MockGenerator()

    def test_compute_free_energy(self):
        expected_free_energy = -7.827218495
        free_energy = self.network.compute_free_energy()
        self.assertAlmostEqual(free_energy, expected_free_energy)

    def test_propagate_signal(self):
        expected_state = np.array([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5])

        self.network.propagate_signal(self.generator)
        network_state = self.network.get_state()
    
        self.assertEqual(len(self.network._state_history_), self.network.state_history_size)
        assert_equal(network_state, expected_state)

        expected_state = np.array([1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

        self.network.propagate_signal(self.generator, {"region1": [1.0, 1.0]})
        network_state = self.network.get_state()

        assert_equal(network_state, expected_state)

        # Test for unknown regions
        with self.assertRaises(NetworkCommunicationError):
            self.network.propagate_signal(self.generator, {"unknown region": [0.0]})

        # Test for mismatching signal
        with self.assertRaises(NetworkCommunicationError):
            self.network.propagate_signal(self.generator, {"region1": [1.0]})

    def test_optimize_connections(self):
        expected_conformation = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                          [0.8, 0.8, np.nan, 0.804, 0.79596, 0.804, 0.804, 0.804, 0.804],
                                          [np.nan, np.nan, 0.804, np.nan, 0.79596, 0.804, 0.804, 0.804, 0.804],
                                          [0.8, 0.8, 0.7866501931,  0.804, np.nan, 0.804, 0.804, 0.804, 0.7866501931],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, np.nan, 0.804, 0.804, 0.804],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, 0.804, np.nan, 0.804, 0.804],
                                          [0.8, 0.8, 0.804, 0.804, 0.79596, 0.804, 0.804, np.nan, 0.804],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, 0.804, 0.804, 0.804, np.nan],])
        
        self.network.propagate_signal(self.generator)
        self.network.optimize_connections()
        conformation = self.network.get_conformation()
        
        assert_equal(np.round(conformation, 7), np.round(expected_conformation, 7))

    def test_remove_neurons(self):
        expected_state = np.array([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0])
        expected_conformation = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                          [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                          [0.8, 0.8, np.nan, 0.804, 0.79596, 0.804, 0.804, 0.804, 0.804],
                                          [np.nan, np.nan, 0.804, np.nan, 0.79596, 0.804, 0.804, 0.804, 0.804],
                                          [0.8, 0.8, 0.7866501931,  0.804, np.nan, 0.804, 0.804, 0.804, 0.7866501931],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, np.nan, 0.804, 0.804, 0.804],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, 0.804, np.nan, 0.804, 0.804],
                                          [0.8, 0.8, 0.804, 0.804, 0.79596, 0.804, 0.804, np.nan, 0.804],
                                          [np.nan, np.nan, 0.804, 0.804, 0.79596, 0.804, 0.804, 0.804, np.nan],])
        

        self.network.propagate_signal(self.generator)
        self.network.remove_neurons(2, "region2", self.generator)
        self.network.optimize_connections()
        network_state = self.network.get_state()
        conformation = self.network.get_conformation()

        assert_equal(network_state, expected_state)
        assert_equal(np.round(conformation, 7), np.round(expected_conformation, 7))
        
    def test_get_conformation(self):
        conformation = self.network.get_conformation()
        self.assertIsInstance(conformation, np.ndarray)

    def test_get_state(self):
        state = self.network.get_state()
        self.assertIsInstance(state, np.ndarray)

    def test_get_internal_conformation(self):
        expected_internal_conformation = np.array([[np.nan, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                                                   [0.8, np.nan, 0.8, 0.8, 0.8, 0.8, 0.8],
                                                   [0.8, 0.8, np.nan, 0.8, 0.8, 0.8, 0.8],
                                                   [0.8, 0.8, 0.8, np.nan, 0.8, 0.8, 0.8],
                                                   [0.8, 0.8, 0.8, 0.8, np.nan, 0.8, 0.8],
                                                   [0.8, 0.8, 0.8, 0.8, 0.8, np.nan, 0.8],
                                                   [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, np.nan],])
        
        internal_conformation = self.network.get_internal_conformation()
        assert_equal(internal_conformation, expected_internal_conformation)

    def test_get_internal_state(self):
        expected_internal_state = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0])
        internal_state = self.network.get_internal_state()
        assert_equal(internal_state, expected_internal_state)

    def test_get_last_internal_state(self):
        expected_last_internal_state = np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5])

        for _ in range(2):
            self.network.propagate_signal(self.generator)

        last_internal_state = self.network.get_last_internal_state()
        assert_equal(last_internal_state, expected_last_internal_state)

    def test_get_motor_signal(self):
        expected_motor_signal = [0.375, 0.392857142]
        self.network.propagate_signal(self.generator)
        motor_signal = self.network.get_motor_signal(("region1", "region2"))
        
        self.assertIsInstance(motor_signal, list)
        self.assertEqual(len(motor_signal), 2)

        for signal, expected_signal in zip(motor_signal, expected_motor_signal):
            self.assertAlmostEqual(signal, expected_signal)

if __name__ == "__main__":
    unittest.main()
