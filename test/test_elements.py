from unittest import TestCase

from simulation.controllers.base_controller import Controller
from simulation.elements.base_element import Element
from simulation.elements.ball import Ball
from simulation.elements.paddle import Paddle
from simulation.geometry.circle import Circle
from simulation.geometry.point import Point

from math import sqrt

class TestController(Controller):
    """Simplified Controller class for testing purposes."""
    speed_increment: Point = Point(0.0, 1.0)

    def __init__(self):
        """Simplified Controller class for testing purposes."""

    def update(self, controlled_element: Element) -> None:
        incremented_speed = controlled_element.speed + self.speed_increment
        controlled_element.set_state(speed=incremented_speed)
        

class TestElements(TestCase):
    
    def test_paddle(self):
        paddle_shape = Circle(center=Point(0.0, 0.0), radius=1.0)
        paddle_controller = TestController()
        paddle = Paddle(shape=paddle_shape, controller=paddle_controller, y_range=(0.0, 4.0))

        expected_positions = [Point(0.0, 1.0), Point(0.0, 3.0), Point(0.0, 4.0)]

        for position in expected_positions:
            paddle.update()

            self.assertEqual(position, paddle.shape.center, msg=f"\n\n{paddle} shape's center does not match {position}")

        with self.assertRaises(ValueError):
            paddle.set_state(position=Point(0.0, -1.0))


    def test_ball(self):
        ball_shape = Circle(center=Point(0.0, 0.0), radius=1.0)
        ball = Ball(shape=ball_shape, speed=Point(1.0, 1.0), acceleration=Point(-1.0, 0.0), speed_range=(0.0, sqrt(2.0)))

        expected_positions = [Point(1.0, 1.0), Point(1.0, 2.0), Point(0.0, 3.0)]
        expected_speeds = [Point(0.0, 1.0), Point(-1.0, 1.0), Point(-1.2649110640, 0.6324555320)]

        for position, speed in zip(expected_positions, expected_speeds):
            expected_shape = Circle(center=position, radius=1.0)
            expected_ball = Ball(shape=expected_shape, speed=speed, acceleration=Point(-1.0, 0.0), speed_range=(0.0, sqrt(2.0)))

            ball.update()

            self.assertEqual(expected_ball, ball, msg=f"\n\n{ball} does not match {expected_ball}")
    
        with self.assertRaises(ValueError):
            ball.set_state(speed=Point(2.0, 2.0))