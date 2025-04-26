import unittest

from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.vector import Vector2D

class TestBaseElement(unittest.TestCase):
    
    def setUp(self):
        main_element_shape = Circle(center=Vector2D(0.0, 0.0), radius=5.0)
        self.main_element = Element(shape=main_element_shape, speed=Vector2D(1.0, -1.0), acceleration=Vector2D(0.0, 1.0))

    def test_element_set_state(self):
        expected_element_shape = Circle(center=Vector2D(0.0, 3.5), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Vector2D(-1.0, 1.0), acceleration=Vector2D(1.0, 0.0))

        self.main_element.set_state(position=Vector2D(0.0, 3.5), speed=Vector2D(-1.0, 1.0), acceleration=Vector2D(1.0, 0.0))
        self.assertEqual(self.main_element, expected_element)

        expected_element_shape = Circle(center=Vector2D(0.0, 3.5), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Vector2D(-10.0, 0.01), acceleration=Vector2D(1.0, 0.0))

        self.main_element.set_state(speed=Vector2D(-10.0, 0.01))
        self.assertEqual(self.main_element, expected_element)

    def test_element_update(self):
        expected_element_shape = Circle(center=Vector2D(1.0, -1.0), radius=5.0)
        expected_element = Element(shape=expected_element_shape, speed=Vector2D(1.0, 0.0), acceleration=Vector2D(0.0, 1.0))

        self.main_element.update()
        self.assertEqual(self.main_element, expected_element)

    def test_element_collides_with(self):
        secondary_element_shape = Circle(center=Vector2D(3.0, 3.0), radius=1.0)
        secondary_element = Element(shape=secondary_element_shape)

        self.assertTrue(self.main_element.collides_with(secondary_element))

if __name__ == "__main__":
    unittest.main()