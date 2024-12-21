import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

    def point_at_parameter(self, t):
        return self.origin + t * self.direction
