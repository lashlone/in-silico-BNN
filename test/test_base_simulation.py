from unittest import TestCase

from simulation.base_simulation import Simulation, load_simulation, load_env_history
from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.triangle import IsoscelesTriangle

import os

class TestBaseElement(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator_seed = 1111

    def setUp(self):
        element_1_shape = Circle(center=Point(0.0, 0.0), radius=10.0)
        element_1 = Element(shape=element_1_shape, speed=Point(1.0, 0.0), acceleration=Point(0.0, 0.25))

        element_2_shape = Rectangle(center=Point(720.0, 400.0), width=10.0, height=10.0, orientation=45.0)
        element_2 = Element(shape=element_2_shape, speed=Point(-2.0, 2.0))

        element_3_shape = IsoscelesTriangle(center=Point(1200.0, 400.0), base=10.0, height=10.0, orientation=180.0)
        element_3 = Element(shape=element_3_shape, speed=Point(0.0, 0.0), acceleration=Point(-1.0, 0.0))

        elements = [element_1, element_2, element_3]
        
        self.simulation = Simulation(height=800, width=1440, frequency=240, elements=elements, simulation_name="test_simulation", generator_seed=TestBaseElement.generator_seed)

    def test_simulation_step(self):
        expected_element_1_shape = Circle(center=Point(10.0, 11.25), radius=10.0)
        expected_element_1 = Element(shape=expected_element_1_shape, speed=Point(1.0, 2.5), acceleration=Point(0.0, 0.25))

        expected_element_2_shape = Rectangle(center=Point(700.0, 420.0), width=10.0, height=10.0, orientation=45.0)
        expected_element_2 = Element(shape=expected_element_2_shape, speed=Point(-2.0, 2.0))

        expected_element_3_shape = IsoscelesTriangle(center=Point(1155.0, 400.0), base=10.0, height=10.0, orientation=180.0)
        expected_element_3 = Element(shape=expected_element_3_shape, speed=Point(-10.0, 0.0), acceleration=Point(-1.0, 0.0))

        expected_elements = [expected_element_1, expected_element_2, expected_element_3]

        for _ in range(10):
            self.simulation.step()

        self.assertEqual(self.simulation._elements, expected_elements)

    def test_simulation_config_file(self):
        # Test save_config function
        expected_config_path = os.path.join("results", "test_simulation", "config.json")

        self.simulation.save_config()

        self.assertTrue(os.path.exists(expected_config_path))

        # Test load_simulation
        expected_loaded_simulation = self.simulation

        loaded_simulation = load_simulation(self.simulation.simulation_name)

        self.assertEqual(loaded_simulation, expected_loaded_simulation)

    def test_simulation_env_history_file(self):
        # Test save_env_history function
        expected_env_history_path = os.path.join("results", "test_simulation", "env_history.json")

        for _ in range(5):
            self.simulation.step()
        self.simulation.save_env_history()

        self.assertTrue(os.path.exists(expected_env_history_path))

        # Test load_env_history
        expected_loaded_env_history = self.simulation._env_history

        loaded_env_history = load_env_history(expected_env_history_path)

        self.assertEqual(loaded_env_history, expected_loaded_env_history)