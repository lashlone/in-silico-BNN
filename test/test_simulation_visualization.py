import unittest

from simulation.base_simulation import load_env_history
from simulation.pong import Pong
from simulation.visualization import generate_gif, generate_success_rate_graph

import numpy as np
import os

class MockPongSimulation(Pong):
    def __init__(self, env_history_data):
        self.simulation_name = "test_simulation"
        self.width = 1440
        self.height = 800
        self._simulation_dir_ = os.path.join("results", self.simulation_name)
        self._env_history_ = env_history_data
        self._success_history_ = [np.array([[1.0 if i % 3 == 2 else 0.0, 10.0 * i],]) for i in range(8)]
        self._timer_ = 200

class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.simulation = MockPongSimulation(load_env_history(os.path.join("results", "test_simulation", "env_history.json")))

    def test_generate_gif(self):
        generate_gif(self.simulation, 25)
        self.assertTrue(os.path.exists(os.path.join("results", "test_simulation", "env_visualization.gif")))

    def test_generate_success_rate_graph(self):
        generate_success_rate_graph(self.simulation)
        self.assertTrue(os.path.exists(os.path.join("results", "test_simulation", "success_rate_evolution.png")))

if __name__ == "__main__":
    unittest.main()