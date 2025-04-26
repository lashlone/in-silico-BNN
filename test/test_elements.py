import unittest

from simulation.controllers.base_controller import Controller
from simulation.elements.ball import Ball
from simulation.elements.base_element import Element
from simulation.elements.paddle import Paddle
from simulation.geometry.circle import Circle
from simulation.geometry.vector import Vector2D

from math import sqrt

class MockController(Controller):
    """Simplified Controller class for testing purposes."""
    speed_increment: Vector2D = Vector2D(0.0, 1.0)

    def __init__(self):
        """Simplified Controller class for testing purposes."""

    def update(self, controlled_element: Element) -> None:
        incremented_speed = controlled_element.speed + self.speed_increment
        controlled_element.set_state(speed=incremented_speed)
        

class TestElements(unittest.TestCase):
    
    def test_paddle(self):
        paddle_shape = Circle(center=Vector2D(0.0, 0.0), radius=1.0)
        paddle_controller = MockController()
        paddle = Paddle(shape=paddle_shape, controller=paddle_controller, y_range=(0.0, 4.0))

        expected_positions = [Vector2D(0.0, 1.0), Vector2D(0.0, 3.0), Vector2D(0.0, 4.0)]

        for expected_position in expected_positions:
            paddle.update()

            self.assertEqual(paddle.shape.center, expected_position, msg=f"\n\n{paddle} shape's center does not match {expected_position}")

        with self.assertRaises(ValueError):
            paddle.set_state(position=Vector2D(0.0, -1.0))


    def test_ball(self):
        ball_shape = Circle(center=Vector2D(0.0, 0.0), radius=1.0)
        ball = Ball(shape=ball_shape, speed=Vector2D(1.0, 1.0), acceleration=Vector2D(-1.0, 0.0), speed_range=(0.0, sqrt(2.0)))

        expected_positions = [Vector2D(1.0, 1.0), Vector2D(1.0, 2.0), Vector2D(0.0, 3.0)]
        expected_speeds = [Vector2D(0.0, 1.0), Vector2D(-1.0, 1.0), Vector2D(-1.2649110640, 0.6324555320)]

        for position, speed in zip(expected_positions, expected_speeds):
            expected_shape = Circle(center=position, radius=1.0)
            expected_ball = Ball(shape=expected_shape, speed=speed, acceleration=Vector2D(-1.0, 0.0), speed_range=(0.0, sqrt(2.0)))

            ball.update()

            self.assertEqual(ball, expected_ball, msg=f"\n\n{ball} does not match {expected_ball}")
    
        with self.assertRaises(ValueError):
            ball.set_state(speed=Vector2D(2.0, 2.0))
            
if __name__ == "__main__":
    unittest.main()