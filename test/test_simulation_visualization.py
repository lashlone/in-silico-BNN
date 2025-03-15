from unittest import TestCase

from simulation.base_simulation import load_env_history
from simulation.visualization import generate_gif

import os

class TestVisualization(TestCase):

    def test_generate_frames(self):
        data = load_env_history(os.path.join("results", "test_simulation", "env_history.json"))

        generate_gif(data, 25)

        self.assertTrue(os.path.exists(os.path.join("results", "test_simulation", "env_visualization.gif")))