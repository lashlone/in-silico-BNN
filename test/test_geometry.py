from unittest import TestCase

from simulation.geometry.circle import Circle
from simulation.geometry.rectangle import Rectangle
from simulation.geometry.triangle import IsoscelesTriangle
from simulation.geometry.point import Point

class TestGeometry(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circle = Circle(center=Point(1.0, 3.0), radius=2.0)
        cls.rectangle = Rectangle(center=Point(3.0, 1.0), width=4.0, height=2.0)
        cls.rotated_rectangle = Rectangle(center=Point(-1.0, 4.0), width=3.5, height=5.0, orientation=30.0)
        cls.triangle = IsoscelesTriangle(center=Point(-2.5, -3.0), base=5.0, height=5.0)
        cls.rotated_triangle = IsoscelesTriangle(center=Point(3.5, -2.0), base=6.0, height=3.0, orientation=225.0)

    def test_rectangle_perimeter(self):
        expected_perimeter = [Point(-0.7345, 7.0401), Point(-3.7655, 5.2901),
                              Point(-1.2655, 0.9599), Point(1.7655, 2.7099)]
        
        result_perimeter = self.rotated_rectangle.get_perimeter_corners()
        result_rounded_perimeter = [point.round(4) for point in result_perimeter]

        self.assertEqual(expected_perimeter, result_rounded_perimeter)
    
    def test_triangle_perimeter(self):
        expected_perimeter = [Point(2.4393, -3.0607),
                              Point(6.6820, -3.0607), Point(2.4393, 1.1820)]

        result_perimeter = self.rotated_triangle.get_perimeter_corners()
        result_rounded_perimeter = [point.round(4) for point in result_perimeter]

        self.assertEqual(expected_perimeter, result_rounded_perimeter)

    def test_circle_contains_point(self):
        # Test a point inside the circle
        self.assertTrue(self.circle.contains_point(Point(2.7213, 3.5263)))

        # Test a point on the circle's perimeter
        self.assertTrue(self.circle.contains_point(Point(1.0, 1.0)))

        # Test a point outside the circle
        self.assertFalse(self.circle.contains_point(Point(-0.4052, 4.5606)))

    def test_rectangle_contains_point(self):
        # Test a point inside the rectangle
        self.assertTrue(self.rotated_rectangle.contains_point(Point(-0.7213, 6.9263)))

        # Test a point on the rectangle's perimeter
        self.assertTrue(self.rotated_rectangle.contains_point(Point(-1.2655, 0.9600)))

        # Test a point outside the rectangle
        self.assertFalse(self.rotated_rectangle.contains_point(Point(-3.5, 4.5)))
    
    def test_triangle_contains_point(self):
        # Test a point inside the triangle
        self.assertTrue(self.triangle.contains_point(Point(-0.1, -3.0)))

        # Test a point on the triangle's perimeter
        self.assertTrue(self.triangle.contains_point(Point(-5.0, -1.8)))

        # Test a point outside the triangle
        self.assertFalse(self.triangle.contains_point(Point(-5.05, -4.6667)))

    def test_circle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.circle.collides_width(Circle(center=Point(-1.9, 2.5), radius=1.0)))

        # Test when only their perimeters overlap
        self.assertTrue(self.circle.collides_width(Circle(center=Point(-4.0, 3.0), radius=3.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.circle.collides_width(Circle(center=Point(-2.0, 3.1), radius=1.0)))

    def test_rectangle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.rectangle.collides_width(Circle(center=Point(0.5, 2.75), radius=1.0)))

        # Test when only their perimeters overlap
        self.assertTrue(self.rectangle.collides_width(Circle(center=Point(2.1, -2.0), radius=2.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.rectangle.collides_width(Circle(center=Point(6.0, -1.8), radius=2.0)))

    def test_triangle_collides_with_circle(self):
        # Test when both shapes overlap
        self.assertTrue(self.rotated_triangle.collides_width(Circle(center=Point(1.05, -1.0), radius=1.5)))

        # Test when only their perimeters overlap
        self.assertTrue(self.rotated_triangle.collides_width(Circle(center=Point(3.5, -4.0606), radius=1.0)))

        # Test when both shapes aren't colliding
        self.assertFalse(self.rotated_triangle.collides_width(Circle(center=Point(2.0, -4.0), radius=1.0)))