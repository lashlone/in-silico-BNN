from unittest import TestCase

from simulation.controllers.pid_controller import VerticalPositionPIDController as VerticalPID
from simulation.elements.base_element import Element
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point

class TestPIDControllers(TestCase):
    
    def test_vertical_PID_controller(self):
        reference_element_shape = Circle(center=Point(5.0, 0.0), radius=1.0)
        reference_element = Element(shape=reference_element_shape, speed=Point(0.0, 1.0))

        controlled_element_shape = Circle(center=Point(0.0, 2.0), radius=1.0)
        controlled_element = Element(shape=controlled_element_shape)

        pid_controller = VerticalPID(kp=0.5, ki=1.0, kd=-0.5, reference=reference_element)

        expected_element_shape = Circle(center=Point(0.0, -1.0), radius=1.0)
        expected_element = Element(shape=expected_element_shape)

        pid_controller.update(controlled_element)
        self.assertEqual(expected_element, controlled_element)

        reference_element.update()
        
        expected_element_shape = Circle(center=Point(0.0, -2.0), radius=1.0)
        expected_element = Element(shape=expected_element_shape)

        pid_controller.update(controlled_element)
        self.assertEqual(expected_element, controlled_element)