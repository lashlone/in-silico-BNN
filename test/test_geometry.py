import unittest

from simulation.geometry.circle import Circle
from simulation.geometry.exceptions import CurvedEdgeError, EdgeError
from simulation.geometry.vector import Vector2D
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.triangle import IsoscelesTriangle

from numpy.random import default_rng

class TestGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circle = Circle(center=Vector2D(1.0, 3.0), radius=2.0)
        cls.rectangle = Rectangle(center=Vector2D(3.0, 1.0), width=4.0, height=2.0)
        cls.rotated_rectangle = Rectangle(center=Vector2D(-1.0, 4.0), width=3.5, height=5.0, orientation=30.0)
        cls.triangle = IsoscelesTriangle(center=Vector2D(-2.5, -3.0), base=5.0, height=5.0)
        cls.rotated_triangle = IsoscelesTriangle(center=Vector2D(3.5, -2.0), base=6.0, height=3.0, orientation=225.0)

    def test_circle_perimeter(self):
        with self.assertRaises(CurvedEdgeError):
            self.circle.get_perimeter_points()

    def test_rectangle_perimeter(self):
        expected_perimeter = [Vector2D(-0.7345, 7.0401), Vector2D(1.7655, 2.7099),
                              Vector2D(-1.2655, 0.9599), Vector2D(-3.7655, 5.2901)]
        
        result_perimeter = self.rotated_rectangle.get_perimeter_points()
        result_rounded_perimeter = [point.round(4) for point in result_perimeter]

        self.assertEqual(result_rounded_perimeter, expected_perimeter)
    
    def test_triangle_perimeter(self):
        expected_perimeter = [Vector2D(2.4393, -3.0607),
                              Vector2D(2.4393, 1.1820), Vector2D(6.6820, -3.0607)]

        result_perimeter = self.rotated_triangle.get_perimeter_points()
        result_rounded_perimeter = [point.round(4) for point in result_perimeter]

        self.assertEqual(result_rounded_perimeter, expected_perimeter)

    def test_circle_contains_point(self):
        # Test a point inside the circle
        self.assertTrue(self.circle.contains_point(Vector2D(2.7213, 3.5263)))

        # Test a point on the circle's perimeter
        self.assertTrue(self.circle.contains_point(Vector2D(1.0, 1.0)))

        # Test a point outside the circle
        self.assertFalse(self.circle.contains_point(Vector2D(-0.4052, 4.5606)))

    def test_rectangle_contains_point(self):
        # Test a point inside the rectangle
        self.assertTrue(self.rotated_rectangle.contains_point(Vector2D(-0.7213, 6.9263)))

        # Test a point on the rectangle's perimeter
        self.assertTrue(self.rotated_rectangle.contains_point(Vector2D(-1.2655, 0.9600)))

        # Test a point outside the rectangle
        self.assertFalse(self.rotated_rectangle.contains_point(Vector2D(-3.5, 4.5)))
    
    def test_triangle_contains_point(self):
        # Test a point inside the triangle
        self.assertTrue(self.triangle.contains_point(Vector2D(-0.1, -3.0)))

        # Test a point on the triangle's perimeter
        self.assertTrue(self.triangle.contains_point(Vector2D(-5.0, -1.8)))

        # Test a point outside the triangle
        self.assertFalse(self.triangle.contains_point(Vector2D(-5.05, -4.6667)))

    def test_circle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.circle.collides_with(Circle(center=Vector2D(-1.9, 2.5), radius=1.0)))

        # Test when only their perimeters overlap
        self.assertTrue(self.circle.collides_with(Circle(center=Vector2D(-4.0, 3.0), radius=3.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.circle.collides_with(Circle(center=Vector2D(-2.0, 3.1), radius=1.0)))

    def test_rectangle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.rectangle.collides_with(Circle(center=Vector2D(0.5, 2.75), radius=1.0)))

        # Test when only their perimeters overlap
        self.assertTrue(self.rectangle.collides_with(Circle(center=Vector2D(2.1, -2.0), radius=2.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.rectangle.collides_with(Circle(center=Vector2D(6.0, -1.8), radius=2.0)))

    def test_triangle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.rotated_triangle.collides_with(Circle(center=Vector2D(1.05, -1.0), radius=1.5)))

        # Test when only their perimeters overlap
        self.assertTrue(self.rotated_triangle.collides_with(Circle(center=Vector2D(3.5, -4.0606), radius=1.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.rotated_triangle.collides_with(Circle(center=Vector2D(2.0, -4.0), radius=1.0)))

    def test_get_random_point(self):
        generator = default_rng()
        # Test that a point generated by the get_random_point function is in the shape 10 times
        for _ in range(10):
            self.assertTrue(self.circle.contains_point(self.circle.get_random_point(generator)))
            self.assertTrue(self.rotated_triangle.contains_point(self.rotated_triangle.get_random_point(generator)))
            self.assertTrue(self.rotated_rectangle.contains_point(self.rotated_rectangle.get_random_point(generator)))

    def test_circle_get_edge_vector(self):
        # Test when the point is on the perimeter of the circle.
        expected_normal_vector = Vector2D(0.7071, 0.7071)
        edge_point = Vector2D(1.41421356237, 1.41421356237)
        result_normal_vector = self.circle.get_edge_normal_vector(edge_point).round(4)
        self.assertEqual(result_normal_vector, expected_normal_vector)

        # Test when the point is off the perimeter of the circle. 
        with self.assertRaises(EdgeError):
            outside_point = Vector2D(1.0, 2.0)
            self.circle.get_edge_normal_vector(outside_point)

    def test_rectangle_get_edge_vector(self):
        # Test when the point is on the perimeter of the rectangle.
        expected_normal_vector = Vector2D(0.0, 1.0)
        edge_point = Vector2D(-0.25, 2.5)
        result_normal_vector = self.rotated_rectangle.get_edge_normal_vector(edge_point).round(4)
        self.assertEqual(result_normal_vector, expected_normal_vector)

        # Test when the point is on one of the corners of the rectangle.
        expected_normal_vector = Vector2D(1.0, 0.0)
        corner_point = Vector2D(1.75, -2.5)
        result_normal_vector = self.rotated_rectangle.get_edge_normal_vector(corner_point).round(4)
        self.assertEqual(result_normal_vector, expected_normal_vector)

        # Test when the point is off the perimeter of the rectangle.
        with self.assertRaises(EdgeError):
            outside_point = Vector2D(1.75, -5.0)
            self.rotated_rectangle.get_edge_normal_vector(outside_point)
    
    def test_triangle_get_edge_vector(self):
        # Test when the point is on the perimeter of the triangle.
        expected_normal_vector = Vector2D(-1.0, 0.0)
        edge_point = Vector2D(-1.5, 1.0)
        result_normal_vector = self.rotated_triangle.get_edge_normal_vector(edge_point).round(4)
        self.assertEqual(result_normal_vector, expected_normal_vector)

        # Test when the point is on one of the corners of the triangle.
        expected_normal_vector = Vector2D(0.7071, 0.7071)
        corner_point = Vector2D(-1.5, 3.0)
        result_normal_vector = self.rotated_triangle.get_edge_normal_vector(corner_point).round(4)
        self.assertEqual(result_normal_vector, expected_normal_vector)

        # Test when the point is off the perimeter of the triangle
        with self.assertRaises(EdgeError):
            outside_point = Vector2D(-1.5, 4.0)
            self.rotated_triangle.get_edge_normal_vector(outside_point)

if __name__ == "__main__":
    unittest.main()