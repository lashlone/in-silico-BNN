from unittest import TestCase

from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point

import numpy as np

class TestBaseElement(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = np.random.default_rng()

    def setUp(self):
        main_element_shape = Circle(center=Point(0.0, 0.0), radius=5.0)
        self.main_element = Element(shape=main_element_shape, speed=Point(1.0, -1.0), acceleration=Point(0.0, 1.0))

    def test_element_set_state(self):
        expected_element_shape = Circle(center=Point(0.0, 3.5), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Point(-1.0, 1.0), acceleration=Point(1.0, 0.0))

        self.main_element.set_state(position=Point(0.0, 3.5), speed=Point(-1.0, 1.0), acceleration=Point(1.0, 0.0))
        self.assertEqual(expected_element, self.main_element)

        expected_element_shape = Circle(center=Point(0.0, 3.5), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Point(-10.0, 0.01), acceleration=Point(1.0, 0.0))

        self.main_element.set_state(speed=Point(-10.0, 0.01))
        self.assertEqual(expected_element, self.main_element)

    def test_element_update(self):
        expected_element_shape = Circle(center=Point(1.0, -1.0), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Point(1.0, 0.0), acceleration=Point(0.0, 1.0))

        self.main_element.update()
        self.assertEqual(expected_element, self.main_element)

    def test_element_collides_with(self):
        secondary_element_shape = Circle(center=Point(3.0, 3.0), radius=1.0)
        secondary_element = Element(shape=secondary_element_shape)

        self.assertTrue(self.main_element.collides_with(secondary_element))