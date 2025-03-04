from unittest import TestCase

from simulation.base_simulation import Simulation
from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.triangle import IsoscelesTriangle

import numpy as np
import os

class TestBaseElement(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = np.random.default_rng()

    def setUp(self):
        element_1_shape = Circle(center=Point(0.0, 0.0), radius=10.0, generator=self.generator)
        element_1 = Element(shape=element_1_shape, speed=Point(1.0, 0.0), acceleration=Point(0.0, 0.25))

        element_2_shape = Rectangle(center=Point(720.0, 400.0), width=10.0, height=10.0, orientation=45.0, generator=self.generator)
        element_2 = Element(shape=element_2_shape, speed=Point(-2.0, 2.0))

        element_3_shape = IsoscelesTriangle(center=Point(1200.0, 400.0), base=10.0, height=10.0, orientation=180.0, generator=self.generator)
        element_3 = Element(shape=element_3_shape, speed=Point(0.0, 0.0), acceleration=Point(-1.0, 0.0))

        elements = [element_1, element_2, element_3]
        
        self.simulation = Simulation(height=800, width=1440, frequency=240, elements=elements, generator=self.generator, simulation_name="test_simulation")

    def test_simulation_update(self):
        expected_element_1_shape = Circle(center=Point(10.0, 11.25), radius=10.0, generator=self.generator)
        expected_element_1 = Element(shape=expected_element_1_shape, speed=Point(1.0, 2.5), acceleration=Point(0.0, 0.25))

        expected_element_2_shape = Rectangle(center=Point(700.0, 420.0), width=10.0, height=10.0, orientation=45.0, generator=self.generator)
        expected_element_2 = Element(shape=expected_element_2_shape, speed=Point(-2.0, 2.0))

        expected_element_3_shape = IsoscelesTriangle(center=Point(1155.0, 400.0), base=10.0, height=10.0, orientation=180.0, generator=self.generator)
        expected_element_3 = Element(shape=expected_element_3_shape, speed=Point(-10.0, 0.0), acceleration=Point(-1.0, 0.0))

        expected_elements = [expected_element_1, expected_element_2, expected_element_3]

        for _ in range(10):
            self.simulation.step()

        self.assertEqual(expected_elements, self.simulation.elements)

    def test_simulation_save_env_history(self):
        expected_env_history_path = os.path.join("results", "test_simulation", "env_history.json")

        for _ in range(5):
            self.simulation.step()
        
        self.simulation.save_env_history_file()

        self.assertTrue(os.path.exists(expected_env_history_path))