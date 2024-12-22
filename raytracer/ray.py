import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32) / np.linalg.norm(direction)

    def at(self, t):
        return self.origin + t * self.direction
