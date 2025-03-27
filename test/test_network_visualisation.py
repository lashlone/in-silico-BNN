import unittest

from network.network import Network

import numpy as np

class MockNetwork(Network):
    def __init__(self):
        self.regions = (
            [1] * 8 +    # Sensory (8 regions of size 1)
            [4] * 8 +    # Afferent (8 regions of size 4)
            [64] +       # Internal (1 region of size 64)
            [12] * 2     # Efferent (2 regions of size 12)
        )

        self._state = np.random.rand(sum(self.regions))
        self._conformation = np.random.uniform(0.0, 1, (len(self._state), len(self._state)))
        self._conformation[np.random.rand(len(self._state), len(self._state)) > 0.1] = np.nan

    def get_state(self):
        return self._state
    
    def get_conformation(self):
        return self._conformation

class TestNetworkVisualization(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()