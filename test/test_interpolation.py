import unittest

from analysis.interpolation import approximate_first_crossing

import numpy as np

class TestApproximateFirstCrossing(unittest.TestCase):

    def test_basic_crossing(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0.5, 0.7, 1.2, 2.0])
        threshold = 1.0

        expected_crossing = 1 + (1.0 - 0.7) * (2 - 1) / (1.2 - 0.7)

        crossing = approximate_first_crossing(x, y, threshold)
        
        self.assertAlmostEqual(expected_crossing, crossing, places=6)

    def test_no_crossing(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0.5, 0.7, 0.9, 0.95])
        threshold = 1.0

        crossing = approximate_first_crossing(x, y, threshold)

        self.assertIsNone(crossing)

    def test_exact_crossing(self):
        x = np.array([0, 1, 2])
        y = np.array([0.5, 1.0, 1.5])
        threshold = 1.0

        crossing = approximate_first_crossing(x, y, threshold)

        self.assertEqual(crossing, 1.0)

    def test_flat_at_threshold(self):
        x = np.array([0, 1, 2])
        y = np.array([1.0, 1.0, 1.0])
        threshold = 1.0

        crossing = approximate_first_crossing(x, y, threshold)

        self.assertIsNone(crossing)

    def test_multiple_crossings(self):
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0.5, 1.2, 0.8, 1.5, 2.0])
        threshold = 1.0
        
        expected_crossing = 0 + (1.0 - 0.5) * (1 - 0) / (1.2 - 0.5)

        crossing = approximate_first_crossing(x, y, threshold)

        self.assertAlmostEqual(crossing, expected_crossing, places=6)

if __name__ == "__main__":
    unittest.main()